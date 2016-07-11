
#pragma once

#include "py_pch.hpp"
#include <iostream>
#include <vector>
#include <annlearn/backprop_net.hpp>

namespace annlearn
{


class base
{
public:
	base() :
		ctx_(vex::Filter::GPU && vex::Filter::Position{0})
	{}

	void reset(const py::object& a)
	{
		std::vector<size_t> l(to_std_vector<size_t>(a));

		std::cout << l.size() << std::endl;
	}

private:
	vex::Context ctx_;
	std::vector<layer<double>> layers_;
};

}
