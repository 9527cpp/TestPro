#include "lj_queue_cache.h"
#include <stdio.h>



int main()
{
	void * qcache = 0;
	qcache = lj_qcache_create(5);
	int a = 0;
	int b = 1;
	int c = 2;
	int d = 3;
	int e = 4;
	lj_qcache_push(qcache, &a);
	lj_qcache_push(qcache, &b);
	lj_qcache_push(qcache, &c);
	lj_qcache_push(qcache, &d);
	lj_qcache_push(qcache, &e);

	//// if all push done  must call this func
	lj_qcache_reset(qcache);
	/// 循环模式
	lj_qcache_set_cycle_mode(qcache);
	/// 线性模式
	lj_qcache_set_line_mode(qcache);

	int * p = NULL;

#if 1
printf("begin get next\r\n");
	for(int i = 0;i<6;i++)
	{
		p = (int *)lj_qcache_get_next(qcache);
		if(p)
		printf("%d\r\n",*p);
		p = NULL;
	}
#endif    
printf("begin get pre\r\n");
	for(int i = 0;i<6;i++)
	{
		p = (int *)lj_qcache_get_pre(qcache);
		if(p)
		printf("%d\r\n",*p);
		p = NULL;
	}

printf("begin get next\r\n");

	for(int i = 0;i<6;i++)
	{
		p = (int *)lj_qcache_get_next(qcache);
		if(p)
		printf("%d\r\n",*p);
		p = NULL;
	}

	return 0;
}
