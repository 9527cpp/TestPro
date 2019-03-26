#include "lj_queue_cache.h"
#include <stdlib.h>
#include <string.h>

typedef struct _lj_qcnode_s {
    void *                  _data;
    struct _lj_qcnode_s * _next;
	struct _lj_qcnode_s * _pre;
} lj_qcnode_t;

enum lj_qmode_t{MODE_LINER = 0,MODE_CYCLE};


typedef struct _lj_qcache_s {
    lj_qcnode_t   *_head;
    lj_qcnode_t   *_tail;
	lj_qcnode_t	  *_cur;
	int			   _max_size;// 0 不限制大小
    int			   _length;
	enum lj_qmode_t	_mode;
} lj_qcache_t;


static lj_qcnode_t *lj_qcache_node_create(void *data)
{
    lj_qcnode_t *node = (lj_qcnode_t *)malloc(sizeof(lj_qcnode_t));
    if (node) {
        node->_data = data;
        node->_next = NULL;
		node->_pre = NULL;
    }
    return node;
}

lj_qcache_handler lj_qcache_create(int max_size)
{
    lj_qcache_t *cache = (lj_qcache_t *)malloc(sizeof(lj_qcache_t));
    if (cache) {
        memset(cache, 0, sizeof(lj_qcache_t));
		cache->_max_size = max_size;
		cache->_mode = MODE_LINER;
    }
/// create head and tail
	cache->_head = lj_qcache_node_create(NULL);
	cache->_tail = lj_qcache_node_create(NULL);
	if(cache->_head)
	{
		cache->_cur = cache->_head;
		cache->_head->_next = cache->_tail;	
	}
	else 
	{
		return NULL;
	}
	if(cache->_tail)
	{
		cache->_tail->_next = NULL;
		cache->_tail->_pre = cache->_head;
	}
	else 
	{
		return NULL;
	}
    return (lj_qcache_handler)cache;
}

int lj_qcache_push(lj_qcache_handler cache, void *data)
{
    int rs = -1;
    lj_qcache_t *p = (lj_qcache_t *)cache;
    lj_qcnode_t *q = NULL;
    do {
        if (p == NULL) {
            break;
        }
        q = lj_qcache_node_create(data);
        if (q == NULL) {
            rs = -1;
            break;
        }
		p->_cur->_next = q;
		q->_pre = p->_cur;
		p->_cur = q;
		p->_cur->_next = p->_tail;	
		p->_tail->_pre = q;
		/// if fixed size
		if(p->_length == p->_max_size && p->_max_size != 0)
		{
			lj_qcnode_t *r = NULL;
			r = p->_head->_next;
			p->_head->_next = r->_next;
			r->_next->_pre = p->_head;
			free(r);
		}
		else
		{
			p->_length++;
		}   
        rs = 0;
    } while (0);
    return rs;
}

int lj_qcache_length(lj_qcache_handler cache)
{
    lj_qcache_t *p = (lj_qcache_t *)cache;
    return (p != NULL) ? p->_length : 0;
}

void *lj_qcache_top(lj_qcache_handler cache)
{
    lj_qcache_t *p = (lj_qcache_t *)cache;
    return p != NULL && p->_head != NULL ? p->_head->_data : NULL;
}

void *lj_qcache_pop(lj_qcache_handler cache)
{
    lj_qcache_t *p = (lj_qcache_t *)cache;
    lj_qcnode_t *q = NULL;
    void *rs = NULL;

    if (p && p->_head) {
        q = p->_head;
        p->_head = q->_next;
        rs = q->_data;
        if (--(p->_length) == 0) {
            p->_tail = NULL;
        }
        free(q);
    }
    return rs;
}

void lj_qcache_reset(lj_qcache_handler cache)
{
	lj_qcache_t *p = (lj_qcache_t *)cache;
	p->_cur = p->_head;
}

void * lj_qcache_get_next(lj_qcache_handler cache)
{
    lj_qcache_t *p = (lj_qcache_t *)cache;
    void *rs = NULL;
	if(p->_cur == p->_head)/// when first
	{		
		p->_cur = p->_head->_next;
		rs = p->_cur->_data;
		return rs;
	}
	if(p->_cur != p->_tail)
	{
		p->_cur = p->_cur->_next;
		rs = p->_cur->_data;
	}
	if(p->_cur == p->_tail)///if ache tail , or be head->next or be tail->pre , so nerver be tail
	{	
		if(p->_mode == MODE_CYCLE)
		{
			p->_cur = p->_head->_next;
			rs = p->_cur->_data;
		}
        else
        {
            p->_cur = p->_tail->_pre;
        }
	}
    return rs;
}

void * lj_qcache_get_pre(lj_qcache_handler cache)
{
    lj_qcache_t *p = (lj_qcache_t *)cache;
    lj_qcnode_t *q = p->_head;
    void *rs = NULL;	
	if(p->_cur == p->_head)//first
	{	
		if(p->_mode == MODE_CYCLE)
		{
			p->_cur = p->_tail->_pre;
			rs = p->_cur->_data;
		}
		else
		{
			p->_cur = p->_head->_next;
        }
		return rs;
	}
    if(p->_cur->_pre == p->_head)
    {
        if(p->_mode == MODE_CYCLE)
        {
            p->_cur = p->_tail;
        }
        else
        {
            return rs;
        }
    }
    p->_cur = p->_cur->_pre;
	rs = p->_cur->_data;
    return rs;
}

void lj_qcache_set_cycle_mode(lj_qcache_handler cache)
{
	lj_qcache_t *p = (lj_qcache_t *)cache;
	p->_mode = MODE_CYCLE;
}

void lj_qcache_set_line_mode(lj_qcache_handler cache)
{
	lj_qcache_t *p = (lj_qcache_t *)cache;
	p->_mode = MODE_LINER;
}
