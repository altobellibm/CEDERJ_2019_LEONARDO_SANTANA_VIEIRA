TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

HEADERS += \
    dual_scaling.h \
    leitorbasenumerica.h

INCLUDEPATH += ./eigen_3.3.4/ \
               C:/boost_1_62_0/

CONFIG += console
