#ifndef _TIMER_H
#define _TIMER_H


#include <ctime>
#include <cstdlib>
#include <string.h>
#include <sys/time.h>

using namespace std;



double inline rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}



class Timer
{
public:
	Timer(const std::string& name): name_ (name),start_ (rtclock())
{
}
	~Timer()
	{
		double elapsed = (rtclock() - start_)*1000;
		cout << name_ << ": " << elapsed << " ms." << endl;
	}
private:
	string name_;
	double start_;
};

#define TIMER(name) Timer timer__(name);


struct timeval now;

#endif
