"""Print the CPU list for the FTX worker threads when FTX_CPUS=auto: the first hardware thread of every physical core
this container may run on (one worker per core). On the measured host (EPYC 7443P, --cpuset-cpus 4-23,28-47, where
28-47 are the SMT siblings of 4-23) this gives 4-23, the campaign's FTX_CPUS."""
import os


def main():
    allowed = sorted(os.sched_getaffinity(0))
    seen, out = set(), []
    for c in allowed:
        try:
            sib = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
        except OSError:
            sib = str(c)
        if sib not in seen:
            seen.add(sib); out.append(c)
    print(",".join(map(str, out)))


if __name__ == "__main__":
    main()
