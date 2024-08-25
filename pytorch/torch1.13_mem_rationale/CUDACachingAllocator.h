/**
 *  The original code: https://github.com/pytorch/pytorch/blob/v1.13.1/c10/cuda/CUDACachingAllocator.h
 *  Modified for independent running.
 *  Author: kevin.xie
 *  Email: kaiyuanxie@yeah.net
 * */
#ifndef THC_DEVICE_ALLOCATOR_INC
#define THC_DEVICE_ALLOCATOR_INC

#include <cuda_runtime_api.h>
// #define EXPANDABLE_SEGMENTS_SUPPORTED 1
#ifdef EXPANDABLE_SEGMENTS_SUPPORTED
#include <cuda.h>
#endif

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cstdio>
#include <deque>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <vector>
#include <optional>

using namespace std;
// original code: using stream_set = ska::flat_hash_set<cuda::CUDAStream>;
// flat_hash_set -> set:
using stream_set = set<cudaStream_t>;

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define C10_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#define C10_UNUSED __attribute__((__unused__))
#else
#define C10_LIKELY(expr) (expr)
#define C10_UNLIKELY(expr) (expr)
#endif

template <typename T> void toStringStream(std::stringstream &ss, T value)
{
    ss << value;
}

template <typename T, typename... Args> void toStringStream(std::stringstream &ss, T first, Args... args)
{
    ss << first;
    toStringStream(ss, args...);
}

template <typename... Args> std::string concatenate(Args... args)
{
    std::stringstream ss;
    toStringStream(ss, args...);
    return ss.str();
}

#define C10_CUDA_CHECK(EXPR)                                                              \
    do {                                                                                  \
        cudaError_t __err = EXPR;                                                         \
        if (__err != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA ERROR: (error code %s)!\n", cudaGetErrorString(__err)); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while (0)

#define C10_CUDA_DRIVER_CHECK(EXPR)                                        \
    do {                                                                     \
        CUresult __err = EXPR;                                                 \
        if (__err != CUDA_SUCCESS) {                                           \
            const char* err_str;                                                 \
            CUresult get_error_str_err C10_UNUSED = cuGetErrorString(__err, &err_str); \
            if (get_error_str_err != CUDA_SUCCESS) {                             \
                printf("CUDA driver error: unknown error\n");                      \
            } else {                                                             \
                printf("CUDA driver error: %s\n", err_str);                          \
            }                                                                    \
        }                                                                      \
    } while (0)

// Simplified torch_check：
#define TORCH_CHECK(cond, ...)                  \
    if (!(cond)) {                              \
        printf("error info:%s", ##__VA_ARGS__); \
        exit(EXIT_FAILURE);                     \
    }

#define TORCH_INTERNAL_ASSERT(...) TORCH_CHECK(__VA_ARGS__)

#define TORCH_CHECK_WITH(cond, ...)               \
    if (!(cond)) {                                \
        cout << concatenate(__VA_ARGS__) << endl; \
        exit(EXIT_FAILURE);                       \
    }

constexpr size_t kMinBlockSize = 512;       // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576;      // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer = 2097152;    // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer = 20971520;   // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc = 10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152;     // round up large allocations to 2 MiB

struct Stat {
    int64_t current = 0;
    int64_t peak = 0;
    int64_t allocated = 0;
    int64_t freed = 0;
};

enum struct StatType : uint64_t {
    AGGREGATE = 0,
    SMALL_POOL = 1,
    LARGE_POOL = 2,
    NUM_TYPES = 3 // remember to update this whenever a new stat type is added
};

using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

typedef std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)> StatArray;

// Struct containing memory allocator summary statistics for a device.
struct DeviceStats {
    // COUNT: allocations requested by client code
    StatArray allocation;
    // COUNT: number of allocated segments from cudaMalloc().
    StatArray segment;
    // COUNT: number of active memory blocks (allocated or used by stream)
    StatArray active;
    // COUNT: number of inactive, split memory blocks (unallocated but can't be
    // released via cudaFree)
    StatArray inactive_split;

    // SUM: bytes requested by client code
    StatArray allocated_bytes;
    // SUM: bytes reserved by this memory allocator (both free and used)
    StatArray reserved_bytes;
    // SUM: bytes within active memory blocks
    StatArray active_bytes;
    // SUM: bytes within inactive, split memory blocks
    StatArray inactive_split_bytes;

    // COUNT: total number of failed calls to CUDA malloc necessitating cache
    // flushes.
    int64_t num_alloc_retries = 0;

    // COUNT: total number of OOMs (i.e. failed calls to CUDA after cache flush)
    int64_t num_ooms = 0;

    // COUNT: total number of oversize blocks allocated from pool
    Stat oversize_allocations;

    // COUNT: total number of oversize blocks requiring malloc
    Stat oversize_segments;

    // SIZE: maximum block size that is allowed to be split.
    int64_t max_split_size = 0;
};

struct Context {
    virtual ~Context()
    {
    }
};

typedef std::unique_ptr<Context> (*CreateContextFn)(void);

struct History {
    void *addr;
    size_t real_size;                 // unrounded, actually requested size
    std::unique_ptr<Context> context; // per-watcher context
    std::unique_ptr<History> next;    // when blocks are merged we keep records of
                                      // what used to be in the block
};

// Struct containing info of an allocation block (i.e. a fractional part of a
// cudaMalloc)..
struct BlockInfo {
    int64_t size = 0;
    int32_t gc_counter = 0;
    bool allocated = false;
    bool active = false;
    History *history = nullptr; // borrowed reference because it is owned by the allocator
};

// Struct containing info of a memory segment (i.e. one contiguous cudaMalloc).
struct SegmentInfo {
    int64_t device = 0;
    int64_t address = 0;
    int64_t total_size = 0;
    int64_t allocated_size = 0;
    int64_t active_size = 0;
    cudaStream_t stream = 0;
    bool is_large = false;
    bool is_expandable = false;
    std::vector<BlockInfo> blocks;
};

struct Block;
struct PrivatePool; // CUDA graphs helper
typedef bool (*Comparison)(const Block *, const Block *);
static bool BlockComparatorSize(const Block* a, const Block* b);
static bool BlockComparatorAddress(const Block* a, const Block* b);

struct BlockPool {
    BlockPool(bool small, PrivatePool *private_pool = nullptr)
        : blocks(BlockComparatorSize)
        , unmapped(BlockComparatorAddress)
        , is_small(small)
        , owner_PrivatePool(private_pool)
    {
    }
    std::set<Block *, Comparison> blocks;
    std::set<Block*, Comparison> unmapped;
    const bool is_small;
    PrivatePool *owner_PrivatePool;
};

struct ExpandableSegment;

struct Block {
    int device;             // gpu
    cudaStream_t stream;    // allocation stream
    stream_set stream_uses; // streams on which the block was used
    size_t size;            // block size in bytes
    BlockPool *pool;        // owning memory pool
    void *ptr;              // memory address
    bool allocated;         // in-use flag
    bool mapped{true};      // is the virtual address range this Block references
                            // backed by physical pages. Always true when
                            // expandable_segment_ is null. When false
                            // This Block will be aligned to the segment size
                            // of its expandable_segment_.
    Block *prev;            // prev block if split from a larger allocation
    Block *next;            // next block if split from a larger allocation
    int event_count;        // number of outstanding CUDA events
    int gc_count;           // counter for prioritizing older / less useful blocks for
                            // garbage collection
    std::unique_ptr<History> history;
    History *history_last;
    ExpandableSegment* expandable_segment_{nullptr};

    Block(int device, cudaStream_t stream, size_t size, BlockPool *pool, void *ptr)
        : device(device)
        , stream(stream)
        , stream_uses()
        , size(size)
        , pool(pool)
        , ptr(ptr)
        , allocated(0)
        , prev(nullptr)
        , next(nullptr)
        , event_count(0)
        , gc_count(0)
    {
    }

    // constructor for search key
    Block(int device, cudaStream_t stream, size_t size)
        : device(device)
        , stream(stream)
        , stream_uses()
        , size(size)
        , pool(nullptr)
        , ptr(nullptr)
        , allocated(0)
        , prev(nullptr)
        , next(nullptr)
        , event_count(0)
        , gc_count(0)
    {
    }

    bool is_split() const
    {
        return (prev != nullptr) || (next != nullptr);
    }

    void splice(Block* before, Block* after) 
    {
        if (before) {
            TORCH_INTERNAL_ASSERT(before->next == after);
            before->next = this;
        }
        prev = before;
        if (after) {
            TORCH_INTERNAL_ASSERT(after->prev == before);
            after->prev = this;
        }
        next = after;
    }

};

struct SegmentRange {
    char* ptr;
    size_t size;
    SegmentRange(void* p, size_t s) : ptr(static_cast<char*>(p)), size(s) {}
};

#ifdef EXPANDABLE_SEGMENTS_SUPPORTED
struct ExpandableSegment {
  ExpandableSegment(
      int device,
      cudaStream_t stream,
      size_t size,
      const std::vector<int>& peers)
      : device_(device),
        stream_(stream),
        max_handles_(0),
        // 2MB for small pool, 20MB for large pool
        segment_size_(size),
        peers_(peers) {
        cudaDeviceProp prop;
        C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_));
        // we allocate enough address space for 1 1/8 the total memory on the GPU.
        // This allows for some cases where we have to unmap pages earlier in the
        // segment to put them at the end.
        max_handles_ = numSegments(prop.totalGlobalMem + prop.totalGlobalMem / 8);
        C10_CUDA_DRIVER_CHECK(cuMemAddressReserve(
        &ptr_, segment_size_ * max_handles_, 0ULL, 0, 0ULL));
    }

    // begin must be aligned to segment_size_.
    // returns the actual range mapped, which may be
    // greater than requested if size is not aligned to segment_size_.
    // return size of 0 indicates OOM
    SegmentRange map(SegmentRange range) {
        auto begin = segmentLeft(range.ptr);
        auto end = segmentRight(range.ptr + range.size);
        TORCH_INTERNAL_ASSERT(ptr() + begin * segment_size_ == range.ptr);
        if (begin == end) 
        {
            return rangeFromHandles(begin, end);
        }
        while (end > handles_.size()) 
        {
            handles_.push_back(std::nullopt);
        }
        // original style: for (int i = begin; i < end; ++i) 
        for (int i = begin; i < end; ++i)
        {
            TORCH_INTERNAL_ASSERT(!handles_.at(i));
            CUmemGenericAllocationHandle handle;
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device_;
            auto status = cuMemCreate(&handle, segment_size_, &prop, 0);
            if (status == CUDA_ERROR_OUT_OF_MEMORY) {
                // original style: for (auto j : c10::irange(begin, i))
                for (int j = begin; j < i; ++j)
                {
                    auto handle = handles_.at(j).value();
                    handles_.at(j) = std::nullopt;
                    C10_CUDA_DRIVER_CHECK(cuMemRelease(handle));
                }
                trimHandles();
                return rangeFromHandles(begin, begin);
            }
            C10_CUDA_DRIVER_CHECK(status);
            handles_.at(i) = handle;
        }
        // original style: for (int i = begin; i < end; ++i)
        for (int i = begin; i < end; ++i)
        {
            C10_CUDA_DRIVER_CHECK(cuMemMap(
                ptr_ + i * segment_size_,
                segment_size_,
                0,
                handles_.at(i).value(),
                0ULL));
        }

        setAccess(device_, begin, end);
        for (auto p : peers_) {
        setAccess(p, begin, end);
        }
        return rangeFromHandles(begin, end);
    }

  // unmaps all the completely empty segment_size_ segments between
  // [begin, begin + size), returns the offset where the range begin,
  // and the actual size unmapped (multiple of segment_size_)
  SegmentRange unmap(SegmentRange range) {
    auto begin = segmentRight(range.ptr);
    auto end = segmentLeft(range.ptr + range.size);
    if (begin >= end) {
      return SegmentRange{range.ptr, 0};
    }
    unmapHandles(begin, end);
    return rangeFromHandles(begin, end);
  }

  char* ptr() const {
    return (char*)ptr_;
  }
  size_t size() const {
    return max_handles_ * segment_size_;
  }

  void addPeer(int device) {
    peers_.push_back(device);
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { setAccess(device, begin, end); });
  }

  ~ExpandableSegment() {
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { unmapHandles(begin, end); });
    C10_CUDA_DRIVER_CHECK(cuMemAddressFree(
        ptr_, segment_size_ * max_handles_));
  }

 private:
  void setAccess(int device, size_t begin, size_t end) {
    CUmemAccessDesc desc;
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = device;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    C10_CUDA_DRIVER_CHECK(cuMemSetAccess(
        ptr_ + begin * segment_size_, (end - begin) * segment_size_, &desc, 1));
  }

  void unmapHandles(size_t begin, size_t end) {
    // note: unlike cudaFree, MemUnmap and MemRelease do
    // not appear to synchronize in all cases, so we have to wait for the
    // stream to finish before this memory is truly free.

    // cannot call c10::cuda::stream_synchronize because
    // it might grab the GIL which can lead to a deadlock
    // Locking order must be GIL -> Allocator Lock
    C10_CUDA_CHECK(cudaStreamSynchronize(stream_));
    for (int i = begin; i < end; ++i) {
      CUmemGenericAllocationHandle h = handles_.at(i).value();
      handles_.at(i) = std::nullopt;
      C10_CUDA_DRIVER_CHECK(cuMemUnmap(
          ptr_ + segment_size_ * i, segment_size_));
      C10_CUDA_DRIVER_CHECK(cuMemRelease(h));
    }
    trimHandles();
  }
  void trimHandles() {
    while (!handles_.empty() && !handles_.back()) {
      handles_.pop_back();
    }
  }
  void forEachAllocatedRange(std::function<void(size_t, size_t)> fn) {
    auto start = 0;
    // for (auto i : c10::irange(handles_.size())) 
    for (int i = 0; i < handles_.size(); ++i)
    {
      if (handles_.at(i) && (i == 0 || !handles_.at(i - 1))) {
        start = i;
      }
      if (handles_.at(i) && (i + 1 == handles_.size() || !handles_.at(i + 1))) {
        fn(start, i + 1);
      }
    }
  }
  size_t numSegments(size_t size) {
    return (size + segment_size_ - 1) / segment_size_;
  }
  size_t segmentLeft(char* p) {
    auto size = p - ptr();
    return size / segment_size_;
  }
  size_t segmentRight(char* p) {
    auto size = p - ptr();
    return numSegments(size);
  }
  SegmentRange rangeFromHandles(size_t begin, size_t end) {
    return SegmentRange(
        ptr() + segment_size_ * begin, segment_size_ * (end - begin));
  }

  int device_;
  cudaStream_t stream_;
  CUdeviceptr ptr_;
  size_t max_handles_;
  size_t segment_size_;
  std::vector<std::optional<CUmemGenericAllocationHandle>> handles_;
  // devices on which this memory should be mapped in addition
  // to the device where the physical memory lives (device_).
  std::vector<int> peers_;

};
#else
struct ExpandableSegment {
  ExpandableSegment(
      int device,
      cudaStream_t stream,
      size_t size,
      const std::vector<int>& peers) {
    TORCH_INTERNAL_ASSERT(false, "expandable segment not supported");
  }
  SegmentRange map(SegmentRange range) {
    return SegmentRange(nullptr, 0);
  }
  SegmentRange unmap(SegmentRange range) {
    return SegmentRange(nullptr, 0);
  }
  char* ptr() const {
    return nullptr;
  }
  size_t size() const {
    return 0;
  }
  void addPeer(int device) {}
};
#endif  // EXPANDABLE_SEGMENTS_SUPPORTED

static bool BlockComparator(const Block *a, const Block *b)
{
    if (a->stream != b->stream) {
        return (uintptr_t)a->stream < (uintptr_t)b->stream;
    }
    if (a->size != b->size) {
        return a->size < b->size;
    }
    return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

static bool BlockComparatorSize(const Block* a, const Block* b)
{
    if (a->stream != b->stream) {
        return (uintptr_t)a->stream < (uintptr_t)b->stream;
    }
    if (a->size != b->size) {
        return a->size < b->size;
    }
    return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

static bool BlockComparatorAddress(const Block* a, const Block* b)
{
    if (a->stream != b->stream) {
        return (uintptr_t)a->stream < (uintptr_t)b->stream;
    }
    return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

struct AllocParams {
    // for set range.
    Block search_key;
    BlockPool *pool;
    size_t alloc_size;
    Block *block;
    StatTypes stat_types = {false};
    cudaError_t err;

    AllocParams(int device, size_t size, cudaStream_t stream, BlockPool *pool, size_t alloc_size, DeviceStats &stats)
        : search_key(device, stream, size)
        , pool(pool)
        , alloc_size(alloc_size)
        , block(nullptr)
        , err(cudaSuccess)
    {
    }

    int device() const
    {
        return search_key.device;
    }
    cudaStream_t stream() const
    {
        return search_key.stream;
    }
    size_t size() const
    {
        return search_key.size;
    }
};

static std::string format_size(uint64_t size);

/* Add some tests */
void testDeviceCachingAllocator();
void testDeviceCachingAllocatorE2E();
void testDeviceCachingAllocatorSmallManagement();
void testDeviceCachingAllocatorFragment();

#endif
