if(NOT TARGET react-native-nitro-image::NitroImage)
add_library(react-native-nitro-image::NitroImage SHARED IMPORTED)
set_target_properties(react-native-nitro-image::NitroImage PROPERTIES
    IMPORTED_LOCATION "C:/Users/harsh/Desktop/Random/NHAI Project/node_modules/react-native-nitro-image/android/build/intermediates/cxx/Debug/45j4r1c1/obj/armeabi-v7a/libNitroImage.so"
    INTERFACE_INCLUDE_DIRECTORIES "C:/Users/harsh/Desktop/Random/NHAI Project/node_modules/react-native-nitro-image/android/build/headers/nitroimage"
    INTERFACE_LINK_LIBRARIES ""
)
endif()

