/*
 Navicat Premium Data Transfer

 Source Server         : se3 mysql
 Source Server Type    : MySQL
 Source Server Version : 80027
 Source Host           : 124.221.127.36:3306
 Source Schema         : springbootjlvpC

 Target Server Type    : MySQL
 Target Server Version : 80027
 File Encoding         : 65001

 Date: 28/05/2022 23:08:32
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for comment
-- ----------------------------

DROP TABLE IF EXISTS `input_helper`;
CREATE TABLE `input_helper`  (
   `ready` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `shape` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `content` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `url` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL
) ENGINE = InnoDB AUTO_INCREMENT = 46 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;
# insert into input_helper values('TRUE', '1, 3, 24, 24', '10.00,9.09,8.223', 'http://127.0.0.1:8080/model.json');

DROP TABLE IF EXISTS `chrome_helper`;
CREATE TABLE `chrome_helper`  (
  `ready` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `type` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `json_total` INT NOT NULL DEFAULT 0,
  `error_content`longtext CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL
) ENGINE = InnoDB AUTO_INCREMENT = 46 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;


DROP TABLE IF EXISTS `msedge_helper`;
CREATE TABLE `msedge_helper`  (
  `ready` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `type` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `json_total` INT NOT NULL DEFAULT 0,
   `error_content`longtext CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL
) ENGINE = InnoDB AUTO_INCREMENT = 46 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;
