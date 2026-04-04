#pragma once

#ifndef __D_EXCEPTION__
#define __D_EXCEPTION__

#include <stdexcept>
#include <string>
using namespace std;

namespace DUtils {

/// General exception
class DException :
	public exception
{
public:
	/**
	 * Creates an exception with a general error message
	 */
	DException(void) throw(): m_message("DUtils exception"){}

	/**
	 * Creates an exception with a custom error message
	 * @param msg: message
	 */
	DException(const char *msg) throw(): m_message(msg){}
	
	/**
	 * Creates an exception with a custom error message
	 * @param msg: message
	 */
	DException(const string &msg) throw(): m_message(msg){}

  /**
	 * Destructor
	 */
	virtual ~DException(void) throw(){}

	/**
	 * Returns the exception message
	 */
	virtual const char* what() const throw()
	{
		return m_message.c_str();
	}

protected:
  /// Error message
	string m_message;
};

}

#endif

