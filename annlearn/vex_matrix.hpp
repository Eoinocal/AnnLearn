
#pragma once


namespace annlearn
{


template<typename T>
struct matrix
{
	matrix() :
		slicer_{vex::extents[1][1]}
	{}

	matrix(size_t col, size_t row, vex::vector<T>&& v) :
		data{std::move(v)},
		col_{col},
		row_{row},
		slicer_{vex::extents[row_][col_]}
	{
		assert(data.size() == col*row);
	}

	matrix(size_t col, size_t row, const vex::vector<T>& v) :
		data{v},
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

	matrix(size_t col, size_t row, std::vector<T>& v) :
		data{v},
		col_{col},
		row_{row},
		slicer_{vex::extents[row_][col_]}
	{
		assert(data.size() == col*row);
	}

	matrix(const std::vector<vex::backend::command_queue>& queue, size_t col, size_t row) :
		data(queue, col*row),
		col_{col},
		row_{row},
		slicer_{vex::extents[row_][col_]}
	{
		assert(data.size() == col*row);
	}

	std::pair<size_t, size_t> dim() const { return std::make_pair(col, row); }

	vex::vector<T> data;

	const auto column(size_t i) const
	{
		return slicer_[vex::_][i](data);
	}

	auto column(size_t i)
	{
		return slicer_[vex::_][i](data);
	}

	const auto row(size_t i) const
	{
		return slicer_[i](data);
	}

	auto row(size_t i)
	{
		return slicer_[i](data);
	}

	const auto slice(const vex::range &row = vex::_, const vex::range &col = vex::_) const
	{
		return slicer_[row][col](data);
	}

	auto slice(const vex::range &row = vex::_, const vex::range &col = vex::_)
	{
		return slicer_[row][col](data);
	}

	void resize(size_t col, size_t row)
	{
		col_ = col;
		row_ = row;

		data.resize(size());
		slicer_ = vex::slicer<2>{vex::extents[nrow()][ncol()]};
	}

	size_t ncol() const { return col_; }
	size_t nrow() const { return row_; }
	size_t size() const { return col_ * row_; }

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & make_nvp("data", data);
		ar & make_nvp("col", col_);
		ar & make_nvp("row", row_);

		slicer_ = vex::extents[row_][col_];
	}

private:
	size_t col_;
	size_t row_;
	vex::slicer<2> slicer_;
};

template<typename T>
matrix<T> load_matrix(std::istream& is)
{
	matrix<T> m;
	boost::archive::xml_iarchive ia(is);
	ia >> make_nvp("m", m);

	return m;
}

}
