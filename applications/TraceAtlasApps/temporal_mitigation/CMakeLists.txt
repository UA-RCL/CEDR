cmake_minimum_required(VERSION 3.2)
project(temporal_mitigation)

set(SOURCE_FILES main.cpp adjoint.cpp alternateinverse.cpp determinant.cpp display.cpp divide.cpp getcofactor.cpp hermitian.cpp imagpart.cpp inverse.cpp mmadd.cpp mmult.cpp mmult4.cpp mmult64.cpp mmultiply.cpp msub.cpp realpart.cpp scalableinverse.cpp)
set(LIBRARIES "m")

add_executable(temporal_mitigation ${SOURCE_FILES})
target_link_libraries(temporal_mitigation ${LIBRARIES})
