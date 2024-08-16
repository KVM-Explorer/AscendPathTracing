#pragma once
#include "common.h"
#include "kernel_operator.h"
#include <memory>

struct MemoryResource {
    uint32_t id;
    uint32_t start;
    uint32_t end;
    bool is_free = true;
    //[start, end)]
    __aicore__ inline MemoryResource() : id(0), start(0), end(0), is_free(true) {}
    __aicore__ inline void Init(uint32_t id, uint32_t start, uint32_t end, bool is_free) {
        this->id = id;
        this->start = start;
        this->end = end;
        this->is_free = is_free;
    }
    __aicore__ inline MemoryResource &operator=(const MemoryResource &other) { return *this; }

    bool operator==(const MemoryResource &other) const { return id == other.id; }
};

struct LinkList {
    MemoryResource data;
    LinkList *next = nullptr;
    LinkList *prev = nullptr;
    __aicore__ inline LinkList() {}
    __aicore__ inline void Init(uint32_t id, uint32_t start, uint32_t end,bool is_free=true) { 
      data.Init(id, start, end, is_free); 
      next = nullptr;
      prev = nullptr;
    }
};

class AllocRet;
class Allocator {
  public:
    /*
     * @brief 临时变量分配器，基于栈的分配器
     * @param base Tensor 起始地址
     * @param length Tensor的最大数据个数
     */
    __aicore__ Allocator(AscendC::LocalTensor<Float> &base, int num);

    /*
     * @brief 申请临时变量Float类型
     * @param size 申请的数据个数
     * @return 临时变量
     */
    __aicore__ AllocRet Alloc(int size);
    __aicore__ void Free(uint32_t id);

  private:
    __aicore__ uint32_t GetNewId() {
        while(true)
        {
            if(Pool[id].data.is_free)
            {
                break;
            }
            id = (id + 1) % MAX_NUM;
        }
        uint32_t ret = id;
        id = (id + 1) % MAX_NUM;
        return ret;
    }

    // insert node after cur
    __aicore__ void InsertNode(LinkList *cur, LinkList *node);
    __aicore__ void DeleteNode(LinkList *cur);
    __aicore__ void MergeNode(LinkList *cur);

    LinkList* head;

    static const uint32_t MAX_NUM{100};
    LinkList Pool[MAX_NUM];

    uint32_t id = 0;
    AscendC::LocalTensor<Float> base;
    int length;
};

class AllocRet {
  public:
    __aicore__ AllocRet(Allocator *allocator, uint32_t id, AscendC::LocalTensor<Float> buf) : allocator(allocator), id(id), buffer(buf) {}
    __aicore__ ~AllocRet();

    AscendC::LocalTensor<Float> buffer;

    __aicore__ const uint32_t GetId() const { return id; }

  private:
    Allocator *allocator;
    uint32_t id;
};
