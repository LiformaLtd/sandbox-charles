#include <libproc.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
  while (1) {
    struct timeval now;
    gettimeofday(&now, NULL);
    printf("%lld", (long long)now.tv_sec * 1000 + now.tv_usec / 1000);
    for (int i = 1; i < argc; i++) {
      struct rusage_info_v4 usage = {0};
      int pid = atoi(argv[i]);
      if (proc_pid_rusage(pid, RUSAGE_INFO_V4, (rusage_info_t *)&usage) != 0)
        printf(" %d:-1:-1", pid);
      else
        printf(" %d:%llu:%llu", pid, usage.ri_phys_footprint, usage.ri_resident_size);
    }
    puts("");
    fflush(stdout);
    usleep(20000);
  }
}
