set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/benches)

add_executable(obliv_poly_eval obliv_poly_eval.cpp)
target_link_libraries(obliv_poly_eval SecureAgg benchmark::benchmark)

add_executable(decryption decryption.cpp)
target_link_libraries(decryption SecureAgg benchmark::benchmark)

add_executable(obliv_share_sample obliv_share_sample.cpp)
target_link_libraries(obliv_share_sample SecureAgg benchmark::benchmark)

add_custom_target(benches)
add_dependencies(benches decryption obliv_poly_eval obliv_share_sample)
