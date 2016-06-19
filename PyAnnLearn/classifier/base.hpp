
#pragma once

#include <iostream>
#include <vector>
#include <annlearn/backprop_net.hpp>

namespace annlearn
{


class base
{
public:
	base()
	{}

	void reset(np::ndarray x)
	{
		try
		{
			if (x.get_dtype() == np::dtype::get_builtin<double>())
				std::cout << "yes" << std::endl;
			else
				std::cout << "no" << std::endl;

		}
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}
	}

	std::vector<layer<double>> layers_;
};

}
