
#include "stdafx.hpp"


boost::unit_test::test_suite* init_unit_test_suite(int argc, char* argv[])
{
	return NULL;
}

int main(int argc, char* argv[])
{
	vex::Context ctx(vex::Filter::GPU && vex::Filter::Position{0});
	if (!ctx) throw std::runtime_error("No devices available.");
	std::cout << ctx << std::endl;

	return ::boost::unit_test::unit_test_main(&init_unit_test_suite, argc, argv);
}
