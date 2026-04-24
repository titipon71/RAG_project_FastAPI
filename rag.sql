-- phpMyAdmin SQL Dump
-- version 5.2.2
-- https://www.phpmyadmin.net/
--
-- Host: db
-- Generation Time: Apr 14, 2026 at 01:59 PM
-- Server version: 9.6.0
-- PHP Version: 8.2.29

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `rag`
--

-- --------------------------------------------------------

--
-- Table structure for table `account_type`
--

CREATE TABLE `account_type` (
  `account_type_id` int NOT NULL,
  `type_name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `file_size_default` bigint DEFAULT NULL COMMENT 'เก็บเป็น byte',
  `created_at` timestamp NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `account_type`
--

INSERT INTO `account_type` (`account_type_id`, `type_name`, `file_size_default`, `created_at`) VALUES
(1, 'personnel', 104857600, '2026-02-22 08:21:50'),
(2, 'student', 52428800, '2026-02-22 08:21:50'),
(3, 'retirement', 157286400, '2026-02-22 08:21:50'),
(4, 'exchange_student', 62914560, '2026-02-22 08:21:50'),
(5, 'alumni', 115343360, '2026-02-22 08:21:50'),
(6, 'guest', 10485760, '2026-02-22 08:21:50');

-- --------------------------------------------------------

--
-- Table structure for table `api_keys`
--

CREATE TABLE `api_keys` (
  `key_id` int UNSIGNED NOT NULL,
  `user_id` int UNSIGNED NOT NULL,
  `channel_id` int UNSIGNED DEFAULT NULL,
  `key_hash` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `is_active` tinyint(1) DEFAULT '1',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `channels`
--

CREATE TABLE `channels` (
  `channels_id` int UNSIGNED NOT NULL,
  `title` varchar(255) NOT NULL,
  `description` text,
  `status` enum('private','public','pending') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT 'private',
  `created_by` int UNSIGNED NOT NULL,
  `search_key` varchar(255) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `channel_status_events`
--

CREATE TABLE `channel_status_events` (
  `event_id` int UNSIGNED NOT NULL,
  `channel_id` int UNSIGNED NOT NULL,
  `old_status` enum('public','private','pending') CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `new_status` enum('public','private','pending') CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `requested_by` int UNSIGNED DEFAULT NULL,
  `decided_by` int UNSIGNED DEFAULT NULL,
  `decision` enum('approved','rejected') CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `decision_reason` varchar(1000) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `is_user_read` tinyint(1) NOT NULL DEFAULT '0',
  `is_admin_read` tinyint(1) NOT NULL DEFAULT '0',
  `soft_delete` tinyint(1) NOT NULL DEFAULT '0',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `decided_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `chats`
--

CREATE TABLE `chats` (
  `chat_id` int UNSIGNED NOT NULL,
  `channels_id` int UNSIGNED NOT NULL,
  `users_id` int UNSIGNED DEFAULT NULL,
  `sessions_id` int UNSIGNED NOT NULL,
  `user_message` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `ai_message` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `files`
--

CREATE TABLE `files` (
  `files_id` int UNSIGNED NOT NULL,
  `uploaded_by` int UNSIGNED DEFAULT NULL,
  `channel_id` int UNSIGNED DEFAULT NULL,
  `original_filename` varchar(512) NOT NULL,
  `storage_uri` varchar(1024) NOT NULL,
  `size_bytes` int UNSIGNED DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `file_size_default`
--

CREATE TABLE `file_size_default` (
  `file_size_default_id` int NOT NULL,
  `size` int NOT NULL COMMENT 'File size MB.'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `sessions`
--

CREATE TABLE `sessions` (
  `sessions_id` int UNSIGNED NOT NULL,
  `channel_id` int UNSIGNED NOT NULL,
  `user_id` int UNSIGNED DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `users_id` int UNSIGNED NOT NULL,
  `username` varchar(255) NOT NULL,
  `name` varchar(255) NOT NULL,
  `hashed_password` varchar(255) NOT NULL,
  `account_type_id` int DEFAULT NULL,
  `file_size_custom` int DEFAULT NULL,
  `role` enum('user','admin','special') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT 'user',
  `active_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `account_type`
--
ALTER TABLE `account_type`
  ADD PRIMARY KEY (`account_type_id`),
  ADD UNIQUE KEY `type_name` (`type_name`);

--
-- Indexes for table `api_keys`
--
ALTER TABLE `api_keys`
  ADD PRIMARY KEY (`key_id`),
  ADD UNIQUE KEY `key_hash` (`key_hash`),
  ADD KEY `user_id` (`user_id`),
  ADD KEY `fk_api_keys_channels` (`channel_id`);

--
-- Indexes for table `channels`
--
ALTER TABLE `channels`
  ADD PRIMARY KEY (`channels_id`),
  ADD KEY `channels_ibfk_1` (`created_by`),
  ADD KEY `search_key` (`search_key`);

--
-- Indexes for table `channel_status_events`
--
ALTER TABLE `channel_status_events`
  ADD PRIMARY KEY (`event_id`),
  ADD KEY `fk_cse_channel` (`channel_id`),
  ADD KEY `fk_cse_requested_by` (`requested_by`),
  ADD KEY `fk_cse_decided_by` (`decided_by`);

--
-- Indexes for table `chats`
--
ALTER TABLE `chats`
  ADD PRIMARY KEY (`chat_id`),
  ADD KEY `fk_chats_users_id` (`users_id`),
  ADD KEY `fk_chats_channels_id` (`channels_id`);

--
-- Indexes for table `files`
--
ALTER TABLE `files`
  ADD PRIMARY KEY (`files_id`),
  ADD KEY `uploaded_by` (`uploaded_by`),
  ADD KEY `idx_files_created_at` (`created_at`),
  ADD KEY `idx_files_channel_id` (`channel_id`);

--
-- Indexes for table `file_size_default`
--
ALTER TABLE `file_size_default`
  ADD PRIMARY KEY (`file_size_default_id`);

--
-- Indexes for table `sessions`
--
ALTER TABLE `sessions`
  ADD PRIMARY KEY (`sessions_id`),
  ADD KEY `channel_id` (`channel_id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`users_id`),
  ADD UNIQUE KEY `name` (`name`),
  ADD KEY `fk_user_account_type` (`account_type_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `account_type`
--
ALTER TABLE `account_type`
  MODIFY `account_type_id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=8;

--
-- AUTO_INCREMENT for table `api_keys`
--
ALTER TABLE `api_keys`
  MODIFY `key_id` int UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `channels`
--
ALTER TABLE `channels`
  MODIFY `channels_id` int UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `channel_status_events`
--
ALTER TABLE `channel_status_events`
  MODIFY `event_id` int UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `chats`
--
ALTER TABLE `chats`
  MODIFY `chat_id` int UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `files`
--
ALTER TABLE `files`
  MODIFY `files_id` int UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `file_size_default`
--
ALTER TABLE `file_size_default`
  MODIFY `file_size_default_id` int NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `sessions`
--
ALTER TABLE `sessions`
  MODIFY `sessions_id` int UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `users_id` int UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `api_keys`
--
ALTER TABLE `api_keys`
  ADD CONSTRAINT `api_keys_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`users_id`),
  ADD CONSTRAINT `fk_api_keys_channels` FOREIGN KEY (`channel_id`) REFERENCES `channels` (`channels_id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `channels`
--
ALTER TABLE `channels`
  ADD CONSTRAINT `channels_ibfk_1` FOREIGN KEY (`created_by`) REFERENCES `users` (`users_id`);

--
-- Constraints for table `channel_status_events`
--
ALTER TABLE `channel_status_events`
  ADD CONSTRAINT `fk_cse_channel` FOREIGN KEY (`channel_id`) REFERENCES `channels` (`channels_id`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `fk_cse_decided_by` FOREIGN KEY (`decided_by`) REFERENCES `users` (`users_id`) ON DELETE SET NULL ON UPDATE CASCADE,
  ADD CONSTRAINT `fk_cse_requested_by` FOREIGN KEY (`requested_by`) REFERENCES `users` (`users_id`) ON DELETE SET NULL ON UPDATE CASCADE;

--
-- Constraints for table `chats`
--
ALTER TABLE `chats`
  ADD CONSTRAINT `fk_chats_channels_id` FOREIGN KEY (`channels_id`) REFERENCES `channels` (`channels_id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  ADD CONSTRAINT `fk_chats_users_id` FOREIGN KEY (`users_id`) REFERENCES `users` (`users_id`);

--
-- Constraints for table `files`
--
ALTER TABLE `files`
  ADD CONSTRAINT `files_ibfk_1` FOREIGN KEY (`uploaded_by`) REFERENCES `users` (`users_id`) ON DELETE SET NULL,
  ADD CONSTRAINT `files_ibfk_2` FOREIGN KEY (`channel_id`) REFERENCES `channels` (`channels_id`) ON DELETE SET NULL;

--
-- Constraints for table `sessions`
--
ALTER TABLE `sessions`
  ADD CONSTRAINT `sessions_ibfk_1` FOREIGN KEY (`channel_id`) REFERENCES `channels` (`channels_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `sessions_ibfk_2` FOREIGN KEY (`user_id`) REFERENCES `users` (`users_id`) ON DELETE CASCADE;

--
-- Constraints for table `users`
--
ALTER TABLE `users`
  ADD CONSTRAINT `fk_user_account_type` FOREIGN KEY (`account_type_id`) REFERENCES `account_type` (`account_type_id`) ON DELETE SET NULL ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
