set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/tests)

add_executable(encryption_test encryption.cpp)
target_link_libraries(encryption_test Boost::unit_test_framework SecureAgg)

add_executable(protocol_test protocol.cpp)
target_link_libraries(protocol_test Boost::unit_test_framework SecureAgg)

add_custom_target(tests)
add_dependencies(tests encryption_test protocol_test)

add_test(NAME encryption_test COMMAND encryption_test)
add_test(NAME protocol_test COMMAND protocol_test)
