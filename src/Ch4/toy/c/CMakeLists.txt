add_mlir_public_c_api_library(ToyCAPI
    toy.cpp
    LINK_LIBS PUBLIC
    MLIRToy
    MLIRParser
    MLIRPass
)
