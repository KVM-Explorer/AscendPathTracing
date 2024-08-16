#pragma once
#include "common.h"
#include "kernel_operator.h"
#include <memory>

struct MemoryResource {
    uint32_t id;
    uint32_t start;
    uint32_t end;
    bool is_free;
    //[start, end)]
    __aicore__ inline MemoryResource(uint32_t id, uint32_t start, uint32_t end) : id(id), start(start), end(end), is_free(true) {}
    __aicore__ inline MemoryResource &operator=(const MemoryResource &other) { return *this; }


    bool operator==(const MemoryResource &other) const { return id == other.id; }
};

struct LinkList {
    MemoryResource data;
    LinkList *next = nullptr;
    LinkList *prev = nullptr;
    __aicore__ inline LinkList(uint32_t id, uint32_t start, uint32_t end) : data(id, start, end), next(nullptr) {}
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
    uint32_t GetNewId() { return id++; }
    uint32_t GetCurId() { return id - 1; }

    // insert node after cur
    void InsertNode(LinkList *cur, LinkList *node);
    void DeleteNode(LinkList *cur);
    void MergeNode(LinkList *cur);

    std::unique_ptr<LinkList> head;

    uint32_t id = 0;
    AscendC::LocalTensor<Float> base;
    int length;
};

class AllocRet {
  public:
    __aicore__ AllocRet(Allocator *allocator, uint32_t id,AscendC::LocalTensor<Float> buf) : allocator(allocator), id(id),buffer(buf) {}
    __aicore__ ~AllocRet();

    AscendC::LocalTensor<Float> buffer;

    const uint32_t GetId() const { return id; }

  private:
    Allocator* allocator;
    uint32_t id;
};
