
#pragma once

#include "stdafx.hpp"
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

	void reset(np::ndarray x)
	{
		vex::vector<double> y{ctx_};

	//	y.r
	}

private:
	vex::Context ctx_;
	std::vector<layer<double>> layers_;
};

}
