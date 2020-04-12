#include "poolman.h"
#include <pthread.h>

extern pthread_mutex_t device_pool_lock;
extern PoolMan device_pool;

extern pthread_mutex_t host_pool_lock;
extern PoolMan host_pool;