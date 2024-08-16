#include "allocator.h"

AllocRet::~AllocRet() {
    if (allocator) {
        allocator->Free(id);
    }
}

Allocator::Allocator(AscendC::LocalTensor<Float> &base, int num) : base(base), length(num) {
    head = std::make_unique<LinkList>(GetNewId(), 0, num);
    // printf("init allocator length %d\n", num);
}

AllocRet Allocator::Alloc(int size) {
    for (auto p = head.get(); p != nullptr; p = p->next) {
        if (p->data.is_free && p->data.end - p->data.start >= size) {

            auto id1 = GetNewId();
            auto id2 = GetNewId();

            if (id1 == 34 && get_block_idx() == 0) {
                int trap = 1;
            }

            // if (get_block_idx() == 0) {

            //     printf("alloc orgin %d %d %d\n", p->data.id, p->data.start, p->data.end);

            //     printf("\t new idx: %d, (%d %d)\n", id1, p->data.start, p->data.start + size);
            //     printf("\t new idx: %d, (%d %d)\n", id2, p->data.start + size, p->data.end);
            // }

            LinkList *res = new LinkList(id2, p->data.start + size, p->data.end);
            InsertNode(p, res);

            p->data.end = p->data.start + size;
            p->data.id = id1;
            p->data.is_free = false;
            return {this, id1,base[p->data.start]};
        }
    }
    throw std::runtime_error("no enough memory");
}

void Allocator::Free(uint32_t id) {

    if (get_block_idx() == 0) {
        int trap = 1;
    }

    // 1. find the memory resource && set the memory resource to free
    bool found = false;
    if (head == nullptr) {
        printf("head is null\n");
    }
    for (auto p = head.get(); p != nullptr; p = p->next) {
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
        auto str = "no such memory resource " + std::to_string(id) + ", latest id is " + std::to_string(GetCurId());
        throw std::runtime_error(str);
    }
    // if (get_block_idx() == 0) {

    //     printf("free %d successfully\n", id);
    // }
}

void Allocator::InsertNode(LinkList *cur, LinkList *node) {
    if (cur->next != nullptr) {
        cur->next->prev = node;
    }
    node->next = cur->next;
    node->prev = cur;
    cur->next = node;
}

void Allocator::DeleteNode(LinkList *cur) {
    if (cur->prev != nullptr) {
        cur->prev->next = cur->next;
    }
    if (cur->next != nullptr) {
        cur->next->prev = cur->prev;
    }
    delete cur;
}

void Allocator::MergeNode(LinkList *cur) {
    if (cur->next != nullptr && cur->next->data.is_free) {
        auto new_id = GetNewId();
        // if (get_block_idx() == 0) {

        //     printf("merge new id: %d current: (%d,%d)\n", new_id, cur->data.start, cur->next->data.end);
        //     printf("\t old cur idx: %d, (%d %d)\n", cur->data.id, cur->data.start, cur->data.end);
        //     printf("\t old nxt idx: %d, (%d %d)\n", cur->next->data.id, cur->next->data.start, cur->next->data.end);
        // }

        cur->data.end = cur->next->data.end;
        DeleteNode(cur->next);
        cur->data.id = new_id;
    }
    if (cur->prev != nullptr && cur->prev->data.is_free) {
        auto new_id = GetNewId();

        // if (get_block_idx() == 0) {

        //     printf("merge new id: %d current: (%d %d)\n", new_id, cur->data.start, cur->prev->data.end);
        //     printf("\t old cur idx: %d,  (%d %d)\n", cur->data.id, cur->prev->data.start, cur->prev->data.end);
        //     printf("\t old pre idx: %d, (%d %d)\n", cur->prev->data.id, cur->data.start, cur->data.end);
        // }

        cur->prev->data.end = cur->data.end;
        DeleteNode(cur);
        cur->prev->data.id = new_id;
    }
}
