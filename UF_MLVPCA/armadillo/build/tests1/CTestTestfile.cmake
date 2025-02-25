# CMake generated Testfile for 
# Source directory: C:/Users/cflee/Documents/0_UFL/PhD/Research/LVPCA/armadillo/tests1
# Build directory: C:/Users/cflee/Documents/0_UFL/PhD/Research/LVPCA/armadillo/build/tests1
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(smoke_test "C:/Users/cflee/Documents/0_UFL/PhD/Research/LVPCA/armadillo/build/tests1/Debug/smoke_test.exe")
  set_tests_properties(smoke_test PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/cflee/Documents/0_UFL/PhD/Research/LVPCA/armadillo/tests1/CMakeLists.txt;5;add_test;C:/Users/cflee/Documents/0_UFL/PhD/Research/LVPCA/armadillo/tests1/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(smoke_test "C:/Users/cflee/Documents/0_UFL/PhD/Research/LVPCA/armadillo/build/tests1/Release/smoke_test.exe")
  set_tests_properties(smoke_test PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/cflee/Documents/0_UFL/PhD/Research/LVPCA/armadillo/tests1/CMakeLists.txt;5;add_test;C:/Users/cflee/Documents/0_UFL/PhD/Research/LVPCA/armadillo/tests1/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(smoke_test "C:/Users/cflee/Documents/0_UFL/PhD/Research/LVPCA/armadillo/build/tests1/MinSizeRel/smoke_test.exe")
  set_tests_properties(smoke_test PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/cflee/Documents/0_UFL/PhD/Research/LVPCA/armadillo/tests1/CMakeLists.txt;5;add_test;C:/Users/cflee/Documents/0_UFL/PhD/Research/LVPCA/armadillo/tests1/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(smoke_test "C:/Users/cflee/Documents/0_UFL/PhD/Research/LVPCA/armadillo/build/tests1/RelWithDebInfo/smoke_test.exe")
  set_tests_properties(smoke_test PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/cflee/Documents/0_UFL/PhD/Research/LVPCA/armadillo/tests1/CMakeLists.txt;5;add_test;C:/Users/cflee/Documents/0_UFL/PhD/Research/LVPCA/armadillo/tests1/CMakeLists.txt;0;")
else()
  add_test(smoke_test NOT_AVAILABLE)
endif()
