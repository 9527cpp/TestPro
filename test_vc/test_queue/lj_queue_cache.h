#ifndef __LJ_QUEUE_CACHE_H__
#define __LJ_QUEUE_CACHE_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef void *lj_qcache_handler;

lj_qcache_handler lj_qcache_create(int max_size);
int lj_qcache_push(lj_qcache_handler cache, void *data);
void *lj_qcache_top(lj_qcache_handler cache);
void *lj_qcache_pop(lj_qcache_handler cache);
int lj_qcache_length(lj_qcache_handler cache);
void lj_qcache_set_cycle_mode(lj_qcache_handler cache);
void lj_qcache_set_line_mode(lj_qcache_handler cache);
void lj_qcache_reset(lj_qcache_handler cache);
void * lj_qcache_get_next(lj_qcache_handler cache);
void * lj_qcache_get_pre(lj_qcache_handler cache);

#ifdef __cplusplus
}
#endif

#endif
