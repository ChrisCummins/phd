# Adding a driver ENUM column to dynamic features.

ALTER TABLE `dynamic_features` ADD COLUMN `driver` enum('CLDRIVE', 'LIBCECL') NOT NULL AFTER `static_features_id`;


UPDATE `dynamic_features`
SET `driver` = 'CLDRIVE';

