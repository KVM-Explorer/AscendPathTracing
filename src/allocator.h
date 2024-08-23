#pragma once
#include "common.h"
#include "kernel_operator.h"

struct MemoryResource {
    uint32_t id;
    uint32_t start;
    uint32_t end;
    bool is_free = true;
    //[start, end)]
    __aicore__ inline MemoryResource() : id(0), start(0), end(0), is_free(true) {}
    __aicore__ void Init(uint32_t id, uint32_t start, uint32_t end, bool is_free) {
        this->id = id;
        this->start = start;
        this->end = end;
        this->is_free = is_free;
    }
    __aicore__ MemoryResource &operator=(const MemoryResource &other) { return *this; }
};

struct LinkList {
    MemoryResource data;
    LinkList *next = nullptr;
    LinkList *prev = nullptr;
    __aicore__ inline LinkList() {}
    __aicore__ void Init(uint32_t id, uint32_t start, uint32_t end, bool is_free = true) {
        data.Init(id, start, end, is_free);
        next = nullptr;
        prev = nullptr;
    }
};

class Allocator;
class AllocDecorator;
class AllocInfo {
  public:
    friend class AllocDecorator;
    __aicore__ inline AllocInfo(Allocator *allocator, uint32_t id, AscendC::LocalTensor<Float> buf) : allocator(allocator), id(id), buffer(buf) {}

    // __aicore__ const uint32_t GetId() const { return id; }

  private:
    AscendC::LocalTensor<Float> buffer;
    Allocator *allocator;
    uint32_t id;
};

class Allocator {
  public:
    /*
     * @brief 临时变量分配器，基于栈的分配器
     * @param base Tensor 起始地址
     * @param length Tensor的最大数据个数
     */
    __aicore__ inline Allocator() {}
    __aicore__ inline void Init(AscendC::LocalTensor<Float> &tensor, int num) {
        base = tensor[0];
        length = num;
        head = &Pool[0];
        head->Init(GetNewId(), 0, num);
        // printf("init allocator length %d\n", num);
    }
    /*
     * @brief 申请临时变量Float类型
     * @param size 申请的数据个数
     * @return 临时变量
     */
    __aicore__ AllocInfo Alloc(int size) {
        for (auto p = head; p != nullptr; p = p->next) {
            if (p->data.is_free && p->data.end - p->data.start >= size) {

                auto id1 = GetNewId();
                auto id2 = GetNewId();

                // if (id1 == 34 && get_block_idx() == 0) {
                //     int trap = 1;
                // }

                // if (get_block_idx() == 0) {

                //     printf("alloc orgin %d (%d %d)\n", p->data.id, p->data.start, p->data.end);

                //     printf("\t new idx: %d, (%d %d)\n", id1, p->data.start, p->data.start + size);
                //     printf("\t new idx: %d, (%d %d)\n", id2, p->data.start + size, p->data.end);
                // }

                LinkList *node2 = &Pool[id2];
                node2->Init(id2, p->data.start + size, p->data.end);
                InsertNode(p, node2);

                LinkList *node1 = &Pool[id1];
                node1->Init(id1, p->data.start, p->data.start + size, false);
                node1->data.is_free = false;
                InsertNode(p, node1);

                DeleteNode(p);
                if (p == head) {
                    head = node1;
                }
                return {this, id1, base[node1->data.start]};
            }
        }
#ifdef ASCENDC_CPU_DEBUG
        throw std::runtime_error("no enough memory");
#endif
        return {nullptr, 0, base[0]};
    }

    __aicore__ void Free(uint32_t id) {

        if (get_block_idx() == 0) {
            int trap = 1;
        }

        // 1. find the memory resource && set the memory resource to free
        bool found = false;
        for (auto p = head; p != nullptr; p = p->next) {
            if (p->data.id == id) {
                found = true;

                // 2. free the memory resource
                if (!p->data.is_free) {
                    p->data.is_free = true;
                    // if (get_block_idx() == 0) {

                    //     printf("free %d %d %d\n", p->data.id, p->data.start, p->data.end);
                    // }
                }
                // else {
                //     printf("warning: idx: %d \n\tmemory resource try to double free (%d,%d)\n", id, p->data.start, p->data.end);
                // }

                // 3. merge the memory resource
                // merge the previous memory resource(keep prev item)
                MergeNode(p);
                break;
            }
        }
        // if (!found && get_block_idx() == 0) {
        //     // printf("no such memory resource %d\n", id);
        //     // for (auto p = head.get(); p != nullptr; p = p->next) {
        //     //     printf("\tid: %d, (%d %d)\n", p->data.id, p->data.start, p->data.end);
        //     // }
        //     auto str = "no such memory resource " + std::to_string(id) + "\n";
        //     throw std::runtime_error(str);
        // }
        // if (get_block_idx() == 0) {

        //     printf("free %d successfully\n", id);
        // }
    }

#ifdef ASCENDC_CPU_DEBUG
    __aicore__ inline bool Check(uint32_t id) {
        for (auto p = head; p != nullptr; p = p->next) {
            if (p->data.id == id) {
                return p->data.is_free;
            }
        }
        throw std::runtime_error("CHECK::no such memory resource");
        return false;
    }
#endif

  private:
    __aicore__ inline bool Contains(uint32_t id) {
        for (auto p = head; p != nullptr; p = p->next) {
            if (p->data.id == id) {
                return true;
            }
        }
        return false;
    }

    __aicore__ uint32_t GetNewId() {
        while (true) {
            if (!Contains(id)) {
                break;
            }
            id = (id + 1) % MAX_NUM;
        }
        uint32_t ret = id;
        id = (id + 1) % MAX_NUM;
        return ret;
    }

    // insert node after cur
    __aicore__ inline void InsertNode(LinkList *cur, LinkList *node) {
        if (cur->next != nullptr) {
            cur->next->prev = node;
        }
        node->next = cur->next;
        node->prev = cur;
        cur->next = node;
    }
    __aicore__ inline void DeleteNode(LinkList *cur) {
        if (cur->prev != nullptr) {
            cur->prev->next = cur->next;
        }
        if (cur->next != nullptr) {
            cur->next->prev = cur->prev;
        }
    }
    __aicore__ void MergeNode(LinkList *cur) {
        uint32_t start = cur->data.start;
        uint32_t end = cur->data.end;
        bool merge = false;
        if (cur->next != nullptr && cur->next->data.is_free) {
            merge = true;
            end = cur->next->data.end;
            // if (get_block_idx() == 0) {

            //     printf("merge new id: %d current: (%d,%d)\n", new_id, cur->data.start, cur->next->data.end);
            //     printf("\t old cur idx: %d, (%d %d)\n", cur->data.id, cur->data.start, cur->data.end);
            //     printf("\t old nxt idx: %d, (%d %d)\n", cur->next->data.id, cur->next->data.start, cur->next->data.end);
            // }
            DeleteNode(cur->next);
        }
        if (cur->prev != nullptr && cur->prev->data.is_free) {
            merge = true;
            start = cur->prev->data.start;
            // if (get_block_idx() == 0) {

            //     printf("merge new id: %d current: (%d %d)\n", new_id, cur->data.start, cur->prev->data.end);
            //     printf("\t old cur idx: %d,  (%d %d)\n", cur->data.id, cur->prev->data.start, cur->prev->data.end);
            //     printf("\t old pre idx: %d, (%d %d)\n", cur->prev->data.id, cur->data.start, cur->data.end);
            // }

            DeleteNode(cur->prev);
        }
        if (merge) {
            auto new_id = GetNewId();
            LinkList *node = &Pool[new_id];
            node->Init(new_id, start, end);
            InsertNode(cur, node);
            DeleteNode(cur);
            if (node->data.start == node->data.end)
                // throw std::runtime_error("merge error");
                AscendC::printf("warning: merge error\n");
        }
    }

    LinkList *head;

    static const uint32_t MAX_NUM{100};
    LinkList Pool[MAX_NUM];

    uint32_t id = 0;
    AscendC::LocalTensor<Float> base;
    int length;
};

class AllocDecorator {
  public:
    __aicore__ inline AllocDecorator(AllocInfo info) : allocInfo(info) {}
    __aicore__ inline ~AllocDecorator() {
        if (!isFree)
            allocInfo.allocator->Free(allocInfo.id);
    }

    __aicore__ inline AscendC::LocalTensor<Float> &Get() {
#ifdef ASCENDC_CPU_DEBUG
        if (allocInfo.allocator->Check(allocInfo.id)) {
            printf("warning: double free %d\n", allocInfo.id);
        }
        if (isFree) {
            // throw std::runtime_error("try to access a free memory");
            AscendC::printf("warning: try to access a free memory %d\n", allocInfo.id);
        }
#endif

        return allocInfo.buffer;
    }
    // __aicore__ const uint32_t GetId() const { return id; }

    __aicore__ inline void Release() {
        if (!isFree) {
            allocInfo.allocator->Free(allocInfo.id);
            isFree = true;
        } else {
            // throw std::runtime_error("try to double free mannually");
            AscendC::printf("warning: double free %d manually\n", allocInfo.id);
        }
    }

  private:
    AllocInfo allocInfo;
    bool isFree = false;
};
