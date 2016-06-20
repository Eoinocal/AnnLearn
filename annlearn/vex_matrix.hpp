
#pragma once


namespace annlearn
{


template<typename T>
struct matrix
{
	matrix(size_t col, size_t row, vex::vector<T>&& v) :
		data{std::move(v)},
		col_{col},
		row_{row},
		slicer_{vex::extents[row_][col_]}
	{
		assert(data.size() == col*row);
	}

	matrix(const std::vector<vex::backend::command_queue>& queue, size_t col, size_t row, std::vector<T>& v) :
		data{queue, v},
		col_{col},
		row_{row},
		slicer_{vex::extents[row_][col_]}
	{
		assert(data.size() == col*row);
	}

	std::pair<size_t, size_t> dim() const { return std::make_pair(col, row); }

	vex::vector<T> data;

	auto column(size_t i) const
	{
		return slicer_[vex::_][i](data);
	}

	auto row(size_t i) const
	{
		return slicer_[i](data);
	}

	auto slice(const vex::range &row = vex::_, const vex::range &col = vex::_) const
	{
		return slicer_[row][col](data);
	}

	size_t ncol() const { return col_; }
	size_t nrow() const { return row_; }
	size_t size() const { return col_ * row_; }

private:
	size_t col_;
	size_t row_;
	vex::slicer<2> slicer_;
};


}
