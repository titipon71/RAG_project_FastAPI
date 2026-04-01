FROM php:8.3.26-apache

RUN docker-php-ext-install mysqli pdo pdo_mysql
