#ifndef YALL_STR_UTILS
#define YALL_STR_UTILS

#include <string>
#include <algorithm>

namespace yall
{
    
    class StringUtils
    {
        public:
            std::string trim(std::string str);
            void trim_all(std::string* strings, int count);
    };

}

#endif
