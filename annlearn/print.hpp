#pragma once

#include "vex_matrix.hpp"

namespace annlearn {

template<typename V>
void print(const V& vs)
{
	for (typename V::value_type v : vs)
		std::cout << (boost::format("%8.5f ") % v);

	std::cout << std::endl << std::endl;
}

template<typename T>
void print(vex::vector<T>& vs)
{
	std::vector<T> us(vs.size());
	vex::copy(vs.begin(), vs.end(), us.begin());
	print(us);
}

template<typename T>
void print(const matrix<T>& vs)
{
	std::vector<T> us(vs.data.size());
	vex::copy(vs.data.begin(), vs.data.end(), us.begin());

	for (int r = 0; r < vs.nrow(); ++r)
	{
		for (size_t c = 0; c < vs.ncol(); ++c)
			std::cout << (boost::format("%8.5f ") % us[r* vs.ncol() + c]);

		std::cout << std::endl;
	}

	std::cout << std::endl;
}

}
