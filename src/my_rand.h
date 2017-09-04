#ifndef MY_RAND_H
#define MY_RAND_H

#include <climits>

#include <iostream>

// Xor-Shift
class my_rand {
private:
	unsigned int s_[4];

	inline unsigned int get_i() {
		const unsigned int t = s_[0] ^ (s_[0] << 11);
		s_[0] = s_[1];
		s_[1] = s_[2];
		s_[2] = s_[3]+1;
		return s_[3] = (s_[3] ^ (s_[3] >> 19)) ^ (t ^ (t >> 8));
	}
public:
	my_rand(unsigned int init = 1234567890) {
		unsigned int s = init;
		s_[0] = s = 1812433253U * (s ^ (s >> 30)) + 1;
		s_[1] = s = 1812433253U * (s ^ (s >> 30)) + 2;
		s_[2] = s = 1812433253U * (s ^ (s >> 30)) + 3;
		s_[3] = s = 1812433253U * (s ^ (s >> 30)) + 4;
	}

	inline double get() {
		uint64_t i = (((uint64_t)get_i() << 20) | (uint64_t)0x3FF0000000000000);
		return (*(double*)&i) - 1.0;
	}
};

#endif // !MY_RAND_H
