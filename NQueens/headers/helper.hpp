#ifndef HELPER_HPP
#define HELPER_HPP

#if defined(__CUDACC__) || defined(__HIPCC__)
    #define CHOP_HD __host__ __device__
#else
    #define CHOP_HD
#endif



double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


#endif
