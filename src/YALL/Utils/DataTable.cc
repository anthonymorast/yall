#include <YALL/Utils/DataTable.hpp>


namespace yall
{
	// TODO: need to seriously test the DataTable class, there is a lot of stuff going on...

	DataTable::DataTable(){}

	DataTable::DataTable(std::string csv_filename, std::string response_name, bool has_headers)
	{
		from_csv(csv_filename, response_name, has_headers);	// set data, ncols, nrows, response vars, and headers in here
		_has_headers = has_headers;
	}

	DataTable::DataTable(std::string* headers, std::string response_name, double** data, int nrows, int ncols, bool has_headers)
	{
		// TODO: should consider deep-copying the pointers in the constructors in case the user frees
		// memory we're using
		_cols = ncols;
		_rows = nrows;
		_data = data;
		_headers = headers;
		_response = response_name;
		_response_column = get_column_from_header(response_name);
		_data_loaded = true;
		_has_headers = has_headers;
	}

	/*! DataTable::DataTable(string*, int, double**, int, int, bool)
	 *  When the response column number is passed into this constructor, it's assumed the user will use
	 *  0-based notation. This is due to the fact that 0-based indexing is used in other get/print methods.
	 */
	DataTable::DataTable(std::string* headers, int response_column, double** data, int nrows, int ncols, bool has_headers)
	{
		_headers = headers;
		_response_column = response_column;
		_response = headers[_response_column];
		_data = data;
		_rows = nrows;
		_cols = ncols;
		_has_headers = has_headers;
		_data_loaded = true;
	}

	DataTable::~DataTable()
	{
		// TODO: ~DataTable(): should probably still free these... will need to pass back new arrays
		// from the getters
		//	delete[] _data;
	}

	bool DataTable::has_response()
	{
		return std::strcmp(_response.c_str(), "");
	}

	int DataTable::get_column_from_header(std::string header)
	{
		int column = _cols + 1;
		for(int i = 0; i < _cols; i++)
		{
			if(_headers[i] == header)
			{
				column = i;
				break;
			}
		}
		if(column == (_cols + 1))
		{
			std::cout << "ERROR: Header '" << header << "' not found." << std::endl;
			return -1;
		}
		return column;
	}

	void DataTable::from_csv(std::string filename, std::string response, bool has_headers)
	{
		if(_data_loaded)
		{
			std::cout << "ERROR: Data has already been loaded for this table." << std::endl;
			return;
		}

		std::ifstream data_file(filename);
		if(!data_file.is_open())
		{
			std::cout << "ERROR: Unable to open file '" << filename << "', no data has been loaded." << std::endl;
			return;
		}

		// read first line, regardless of headers, to get valuable data (ncols)
		std::string line;
		std::getline(data_file, line);
		std::istringstream ss(line);
		std::string value;
		_cols = 0;
		while(std::getline(ss, value, ','))
		{
			_cols++;
		}

		if(has_headers)		// read the header line if there is one
		{
			_headers = new std::string[_cols];
			std::istringstream ss_headers(line);
			bool response_found = false;
			int count = 0;
			while(std::getline(ss_headers, value, ','))
			{
				if(!response_found && !std::strcmp(value.c_str(), response.c_str()))
				{
					response_found = true;
					_response = value;
					_response_column = count;
				}
				_headers[count] = value;
				count++;
			}
			if(!response_found)
			{
				std::cout << "ERROR: response variable not found." << std::endl;
				return;
			}
		}

		// process the remainder of the data file; first need to get rowcount to allocate memory
		_rows = 1;	// first line above
		while(std::getline(data_file, line))
		{
			_rows++;
		}
		_data = new double*[_rows];
		for(int i = 0; i < _rows; i++)
			_data[i] = new double[_cols];

		data_file.clear();	// clear eof flag
		data_file.seekg(0, std::ios::beg);	// go to beginning of file
		if(has_headers)
			std::getline(data_file, line); // skip headers

		int row_count = 0;
		while(std::getline(data_file, line)) 
		{
			int col_count = 0;
			std::istringstream ss2(line);
			while(std::getline(ss2, value, ','))
			{
				try
				{
					_data[row_count][col_count] = std::stod(value);
				}
				catch(std::exception &e)
				{
					std::cout << "ERROR: value '" << value << "' is not parsable as a double." << std::endl;
					return;
				}
				col_count++;
			}
			if(col_count != _cols)
			{
				std::cout << "ERROR: too many columns in row " << (row_count+1) << "." << std::endl;
			}
			row_count++;
		}

		data_file.close();
		_data_loaded = true;
	}

	void DataTable::to_file(std::string filename, char delimiter)
	{
		std::ofstream out(filename);
		if(!out.is_open())
		{
			std::cout << "ERROR: could not open file '" << filename << "'." << std::endl;
			return;
		}

		if(!_data_loaded)
		{
			std::cout << "ERROR: writing empty data table." << std::endl;
			out.close();
			return;
		}

		if(_has_headers)
		{
			for(int i = 0; i < _cols; i++)
			{
				out << _headers[i];
				if(i < _cols-1)
					out << delimiter;
			}
			out << std::endl;
		}

		for(int i = (_has_headers ? 1 : 0); i < _rows; i++)
		{
			for(int j = 0; j < _cols; j++)
			{
				out << _data[i][j];
				if(j < _cols-1)
					out << delimiter;
			}
			out << std::endl;
		}
	}

	double** DataTable::get_data() { return _data; }
	double* DataTable::get_row(int row) 
	{ 
		if(row >= _rows || row < 0)
		{
			std::cout << "ERROR: row index out of range; row " << row << " out of " << (_rows+1) << std::endl;
			std::cout << "NOTE: row indexing is 0-based." << std::endl;
			return 0;
		}
		return _data[row]; 
	}

	double* DataTable::get_column(int column) 
	{ 	
		if(column >= _cols || column < 0)
		{
			std::cout << "ERROR: column index out of range; column " << column << " out of " << (_cols+1) << std::endl;
			std::cout << "NOTE: column indexing is 0-based." << std::endl;
			return 0;
		}

		double* col_data = new double[_rows];
		for(int i = 0; i < _rows; i++)
		{
			col_data[i] = _data[i][column];
		}
		return col_data; 
	}

	double* DataTable::get_column(std::string column_name)
	{
		int col = get_column_from_header(column_name);
		return (col < 0 ? 0 : get_column(col));
	}

	double* DataTable::get_response()
	{
		if(!std::strcmp(_response.c_str(), ""))
		{
			std::cout << "ERROR: response variable not set." << std::endl;
			return 0;
		}
		int col = get_column_from_header(_response);
		return (col < 0 ? 0 : get_column(_response));
	}

	double** DataTable::get_all_explanatory()
	{
		double** data = new double*[_rows];
		for(int i = 0; i < _rows; i++)
			data[i] = new double[_cols-1];

		for(int i = 0; i < _rows; i++)
		{
			for(int j = 0; j < _cols; j++)
			{
				// TODO: consider using the trick where the response is moved to the first column (or last)
				if(j < _response_column)
					data[i][j] = _data[i][j];
				else if(j > _response_column)	// use previous index after response column
					data[i][j-1] = _data[i][j];
			}
		}
		return data;
	}

	void DataTable::print(std::ostream& stream)
	{	
		if(!_data_loaded)
		{
			std::cout << "ERROR: printing empty data table." << std::endl;
			return;
		}

		char delimiter = ',';
		if(_has_headers)
		{
			for(int i = 0; i < _cols; i++)
			{
				stream << _headers[i];
				if(i < _cols-1)
					stream << delimiter;
			}
			stream << std::endl;
		}

		for(int i = (_has_headers ? 1 : 0); i < _rows; i++)
		{
			for(int j = 0; j < _cols; j++)
			{
				stream << _data[i][j];
				if(j < _cols-1)
					stream << delimiter;
			}
			stream << std::endl;
		}
	}

	void DataTable::print_shape(std::ostream& stream)
	{
		int* s = shape();
		stream << "(" << s[0] << ", " << s[1] << ")" << std::endl;
	}

	void DataTable::print_column(std::ostream& stream, int column)
	{
		if(column >= _cols || column < 0)
		{
			std::cout << "ERROR: column index out of range; column " << column << " out of " << (_cols+1) << std::endl;
			std::cout << "NOTE: column indexing is 0-based." << std::endl;
			return;
		}

		for(int i = 0; i < _rows; i++)
			stream << _data[i][column] << std::endl;
	}

	void DataTable::print_column(std::ostream& stream, std::string column)
	{
		int	col = get_column_from_header(column);
		if(col < 0)
			return;
		print_column(stream, col);
	}

	void DataTable::print_row(std::ostream& stream, int row)
	{	
		if(row >= _rows || row < 0)
		{
			std::cout << "ERROR: row index out of range; row " << row << " out of " << (_rows+1) << std::endl;
			std::cout << "NOTE: row indexing is 0-based." << std::endl;
			return;
		}

		for(int i = 0; i < _cols; i++)
			stream << _data[row][i] << ", ";
		stream << std::endl;
	}

	void DataTable::print_headers(std::ostream& stream)
	{
		for(int i = 0; i < _cols; i++) 
			stream << _headers[i] << (i < _cols-1 ? ", " : "");
		stream << std::endl;
	}

	double* DataTable::operator[](int row) const
	{
		if(row >= _rows || row < 0)
		{
			std::cout << "ERROR: row index out of range; row " << row << " out of " << (_rows+1) << std::endl;
			std::cout << "NOTE: row indexing is 0-based." << std::endl;
			return 0;
		}
		return _data[row]; 
	}

	std::ostream& operator<<(std::ostream& os, const DataTable& table)
	{
		if(!table._data_loaded)
		{
			os << "ERROR: printing empty data table." << std::endl;
			return os;
		}

		char delimiter = ',';
		if(table._has_headers)
		{
			for(int i = 0; i < table._cols; i++)
			{
				os << table._headers[i];
				if(i < table._cols-1)
					os << delimiter;
			}
			os << std::endl;
		}

		for(int i = (table._has_headers ? 1 : 0); i < table._rows; i++)
		{
			for(int j = 0; j < table._cols; j++)
			{
				os << table._data[i][j];
				if(j < table._cols-1)
					os << delimiter;
			}
			os << std::endl;
		}

		return os;
	}

}
