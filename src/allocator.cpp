#include "allocator.h"

__aicore__ AllocRet::~AllocRet() {
    if (allocator) {
        allocator->Free(id);
    }
}

__aicore__ Allocator::Allocator(AscendC::LocalTensor<Float> &base, int num) : base(base), length(num) {
    head = &Pool[0];
    head->Init(GetNewId(), 0, num);
    // printf("init allocator length %d\n", num);
}

__aicore__ AllocRet Allocator::Alloc(int size) {
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
            node1->Init(id1, p->data.start, p->data.start + size,false);
            node1->data.is_free = false;
            InsertNode(p, node1);

            DeleteNode(p);
            if(p == head){
                head = node1;
            }
            return {this, id1,base[node1->data.start]};
        }
    }
    throw std::runtime_error("no enough memory");
}

__aicore__ void Allocator::Free(uint32_t id) {

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
    if (!found && get_block_idx() == 0) {
        // printf("no such memory resource %d\n", id);
        // for (auto p = head.get(); p != nullptr; p = p->next) {
        //     printf("\tid: %d, (%d %d)\n", p->data.id, p->data.start, p->data.end);
        // }
        auto str = "no such memory resource " + std::to_string(id) + "\n";
        throw std::runtime_error(str);
    }
    // if (get_block_idx() == 0) {

    //     printf("free %d successfully\n", id);
    // }
}

__aicore__ void Allocator::InsertNode(LinkList *cur, LinkList *node) {
    if (cur->next != nullptr) {
        cur->next->prev = node;
    }
    node->next = cur->next;
    node->prev = cur;
    cur->next = node;
}

__aicore__ void Allocator::DeleteNode(LinkList *cur) {
    if (cur->prev != nullptr) {
        cur->prev->next = cur->next;
    }
    if (cur->next != nullptr) {
        cur->next->prev = cur->prev;
    }
}

__aicore__ void Allocator::MergeNode(LinkList *cur) {
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
    if(merge)
    {
        auto new_id = GetNewId();
        LinkList *node = &Pool[new_id];
        node->Init(new_id, start, end);
        InsertNode(cur, node);
        DeleteNode(cur);
    }
}
