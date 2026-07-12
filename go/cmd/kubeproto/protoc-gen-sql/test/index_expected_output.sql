CREATE TABLE `test_indexing`
(
    `uid`         VARCHAR(255) NOT NULL,
    `group_ver`   VARCHAR(255) NOT NULL,
    `namespace`   VARCHAR(255) NOT NULL,
    `name`        VARCHAR(255) NOT NULL,
    `res_version` BIGINT UNSIGNED NOT NULL,
    `create_time` DATETIME     NOT NULL,
    `update_time` DATETIME,
    `delete_time` DATETIME,
    `proto`       MEDIUMBLOB,
    `json`        JSON,
    `key01`    VARCHAR(255),
    `key02`    INT,
    `key03`    INT,
    `key04`    BIGINT,
    `key05_namespace`    VARCHAR(255),
    `key05_name`    VARCHAR(255),
    `key06_name`    VARCHAR(255),
    `key06_proxy_user`    VARCHAR(255),
    `key07`    VARCHAR(255),
    `key08`    VARCHAR(255),
    `key09`    BIGINT,
    `key10`    INT,
    `key11`    VARCHAR(255),
    `key12`    BIGINT,
    `key13`    DATETIME,
    `key14`    DATETIME,
    `key15`    VARCHAR(255),
    `key16`    BOOLEAN,
    `key17`    BOOLEAN,
    `key18`    VARCHAR(255),
    `key19`    VARCHAR(768),
    `key20`    VARCHAR(255),
    `key21`    VARCHAR(768),
    PRIMARY KEY   (`uid`),
    KEY    `test_indexing_namespace_name` (`namespace`, `name`),
    KEY    `test_indexing_create_time` (`create_time`),
    KEY    `test_indexing_update_time` (`update_time`),
    KEY    `test_indexing_delete_time` (`delete_time`),
    KEY    `test_indexing_namespace_timestamp` (`namespace`, `delete_time`, `create_time`, `update_time`),
    KEY    `test_indexing_key01` (`key01`),
    KEY    `test_indexing_key02` (`key02`),
    KEY    `test_indexing_key03` (`key03`),
    KEY    `test_indexing_key04` (`key04`),
    KEY    `test_indexing_key05` (`key05_namespace`, `key05_name`),
    KEY    `test_indexing_key06_name` (`key06_name`),
    KEY    `test_indexing_key06_proxy_user` (`key06_proxy_user`),
    KEY    `test_indexing_key07` (`key07`),
    KEY    `test_indexing_key08` (`key08`),
    KEY    `test_indexing_key09` (`key09`),
    KEY    `test_indexing_key10` (`key10`),
    KEY    `test_indexing_key11` (`key11`),
    KEY    `test_indexing_key12` (`key12`),
    KEY    `test_indexing_key13` (`key13`),
    KEY    `test_indexing_key14` (`key14`),
    KEY    `test_indexing_key15` (`key15`),
    KEY    `test_indexing_key16` (`key16`),
    KEY    `test_indexing_key17` (`key17`),
    KEY    `test_indexing_key18` (`key18`),
    KEY    `test_indexing_key19` (`key19`),
    KEY    `test_indexing_key20` (`key20`),
    KEY    `test_indexing_key21` (`key21`)
);
CREATE TABLE `test_indexing_labels`
(
    `id`      BIGINT       NOT NULL AUTO_INCREMENT,
    `obj_uid` VARCHAR(255) NOT NULL,
    `key`     VARCHAR(255) NOT NULL,
    `value`   VARCHAR(63),
    PRIMARY KEY (`id`),
    KEY    `test_indexing_labels_uid` (`obj_uid`),
    KEY    `test_indexing_labels_value` (`key`, `value`)
);
CREATE TABLE `test_indexing_annotations`
(
    `id`      BIGINT       NOT NULL AUTO_INCREMENT,
    `obj_uid` VARCHAR(255) NOT NULL,
    `key`     VARCHAR(255) NOT NULL,
    `value`   TEXT,
    PRIMARY KEY (`id`),
    KEY    `test_indexing_annotations_uid` (`obj_uid`)
);
CREATE TABLE `test_base`
(
    `uid`         VARCHAR(255) NOT NULL,
    `group_ver`   VARCHAR(255) NOT NULL,
    `namespace`   VARCHAR(255) NOT NULL,
    `name`        VARCHAR(255) NOT NULL,
    `res_version` BIGINT UNSIGNED NOT NULL,
    `create_time` DATETIME     NOT NULL,
    `update_time` DATETIME,
    `delete_time` DATETIME,
    `proto`       MEDIUMBLOB,
    `json`        JSON,
    `test_name`    VARCHAR(255),
    `test_ref_namespace`    VARCHAR(255),
    `test_ref_name`    VARCHAR(255),
    PRIMARY KEY   (`uid`),
    KEY    `test_base_namespace_name` (`namespace`, `name`),
    KEY    `test_base_create_time` (`create_time`),
    KEY    `test_base_update_time` (`update_time`),
    KEY    `test_base_delete_time` (`delete_time`),
    KEY    `test_base_namespace_timestamp` (`namespace`, `delete_time`, `create_time`, `update_time`),
    KEY    `test_base_test_name` (`test_name`),
    KEY    `test_base_test_ref` (`test_ref_namespace`, `test_ref_name`)
);
CREATE TABLE `test_base_labels`
(
    `id`      BIGINT       NOT NULL AUTO_INCREMENT,
    `obj_uid` VARCHAR(255) NOT NULL,
    `key`     VARCHAR(255) NOT NULL,
    `value`   VARCHAR(63),
    PRIMARY KEY (`id`),
    KEY    `test_base_labels_uid` (`obj_uid`),
    KEY    `test_base_labels_value` (`key`, `value`)
);
CREATE TABLE `test_base_annotations`
(
    `id`      BIGINT       NOT NULL AUTO_INCREMENT,
    `obj_uid` VARCHAR(255) NOT NULL,
    `key`     VARCHAR(255) NOT NULL,
    `value`   TEXT,
    PRIMARY KEY (`id`),
    KEY    `test_base_annotations_uid` (`obj_uid`)
);
CREATE TABLE `test_base_test_wrapper_unmarshalled`
(
    `test_wrapper_uid` VARCHAR(255) NOT NULL,
    `test_name`    VARCHAR(255),
    `test_ref_namespace`    VARCHAR(255),
    `test_ref_name`    VARCHAR(255),
    PRIMARY KEY (`test_wrapper_uid`),
    KEY    `test_base_test_wrapper_unmarshalled_test_name` (`test_name`),
    KEY    `test_base_test_wrapper_unmarshalled_test_ref_namespace` (`test_ref_namespace`),
    KEY    `test_base_test_wrapper_unmarshalled_test_ref_name` (`test_ref_name`)
);
CREATE TABLE `test_wrapper`
(
    `uid`         VARCHAR(255) NOT NULL,
    `group_ver`   VARCHAR(255) NOT NULL,
    `namespace`   VARCHAR(255) NOT NULL,
    `name`        VARCHAR(255) NOT NULL,
    `res_version` BIGINT UNSIGNED NOT NULL,
    `create_time` DATETIME     NOT NULL,
    `update_time` DATETIME,
    `delete_time` DATETIME,
    `proto`       MEDIUMBLOB,
    `json`        JSON,
    PRIMARY KEY   (`uid`),
    KEY    `test_wrapper_namespace_name` (`namespace`, `name`),
    KEY    `test_wrapper_create_time` (`create_time`),
    KEY    `test_wrapper_update_time` (`update_time`),
    KEY    `test_wrapper_delete_time` (`delete_time`),
    KEY    `test_wrapper_namespace_timestamp` (`namespace`, `delete_time`, `create_time`, `update_time`)
);
CREATE TABLE `test_wrapper_labels`
(
    `id`      BIGINT       NOT NULL AUTO_INCREMENT,
    `obj_uid` VARCHAR(255) NOT NULL,
    `key`     VARCHAR(255) NOT NULL,
    `value`   VARCHAR(63),
    PRIMARY KEY (`id`),
    KEY    `test_wrapper_labels_uid` (`obj_uid`),
    KEY    `test_wrapper_labels_value` (`key`, `value`)
);
CREATE TABLE `test_wrapper_annotations`
(
    `id`      BIGINT       NOT NULL AUTO_INCREMENT,
    `obj_uid` VARCHAR(255) NOT NULL,
    `key`     VARCHAR(255) NOT NULL,
    `value`   TEXT,
    PRIMARY KEY (`id`),
    KEY    `test_wrapper_annotations_uid` (`obj_uid`)
);
CREATE TABLE `test_draft`
(
    `uid`         VARCHAR(255) NOT NULL,
    `group_ver`   VARCHAR(255) NOT NULL,
    `namespace`   VARCHAR(255) NOT NULL,
    `name`        VARCHAR(255) NOT NULL,
    `res_version` BIGINT UNSIGNED NOT NULL,
    `create_time` DATETIME     NOT NULL,
    `update_time` DATETIME,
    `delete_time` DATETIME,
    `proto`       MEDIUMBLOB,
    `json`        JSON,
    PRIMARY KEY   (`uid`),
    KEY    `test_draft_namespace_name` (`namespace`, `name`),
    KEY    `test_draft_create_time` (`create_time`),
    KEY    `test_draft_update_time` (`update_time`),
    KEY    `test_draft_delete_time` (`delete_time`),
    KEY    `test_draft_namespace_timestamp` (`namespace`, `delete_time`, `create_time`, `update_time`)
);
CREATE TABLE `test_draft_labels`
(
    `id`      BIGINT       NOT NULL AUTO_INCREMENT,
    `obj_uid` VARCHAR(255) NOT NULL,
    `key`     VARCHAR(255) NOT NULL,
    `value`   VARCHAR(63),
    PRIMARY KEY (`id`),
    KEY    `test_draft_labels_uid` (`obj_uid`),
    KEY    `test_draft_labels_value` (`key`, `value`)
);
CREATE TABLE `test_draft_annotations`
(
    `id`      BIGINT       NOT NULL AUTO_INCREMENT,
    `obj_uid` VARCHAR(255) NOT NULL,
    `key`     VARCHAR(255) NOT NULL,
    `value`   TEXT,
    PRIMARY KEY (`id`),
    KEY    `test_draft_annotations_uid` (`obj_uid`)
);
