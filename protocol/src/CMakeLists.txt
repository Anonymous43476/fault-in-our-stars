add_library(SecureAgg
    secureagg/encryption.cpp
    secureagg/obliv_share_sample.cpp)
target_include_directories(SecureAgg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SecureAgg PUBLIC NTL GMP OpenSSL::Crypto)
