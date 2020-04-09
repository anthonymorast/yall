#include <vector>
#include <iostream>

using namespace std;

class Base 
{
	public:
		virtual void say_hi() { cout << "hi from base" << endl;	}
};

class Derived: public Base
{
	public:
		virtual void say_hi() { cout << "hi from derived" << endl; }
};

int main()
{
	Base *b = new Derived;
	Base a;

	a.say_hi();
	b->say_hi();
}
