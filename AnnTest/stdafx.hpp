
#pragma once

#include <stdio.h>
#include <tchar.h>

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#pragma warning (disable : 4003)

#pragma warning (push)
//#define VEXCL_SHOW_KERNELS
#pragma warning (disable : 4996)
#include <vexcl/vexcl.hpp>
#include <vexcl/reductor.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/tensordot.hpp>
#pragma warning (pop)
