create table aws_instances
(
    id            VARCHAR(32) not null
        primary key
        unique,
    type          VARCHAR(32),
    public_ip_v4  VARCHAR(16),
    state         VARCHAR(32),
    job_reference VARCHAR(64)
);

create table processes
(
    reference      VARCHAR(64) not null
        primary key,
    progress FLOAT   not null
);

create table users
(
    id       INTEGER      not null
        primary key auto_increment,
    username VARCHAR(255) not null,
    api_key  VARCHAR(64)  not null
        unique
);

create table finetune_jobs
(
    id              VARCHAR(36)  not null
        primary key,
    model_reference VARCHAR(255) not null,
    gender          VARCHAR(16)  not null,
    max_train_steps INTEGER      not null,
    user_id         INTEGER
        references users,
    aws_instance_id VARCHAR(32)
        references aws_instances,
    unique (user_id, model_reference)
);

create table finetune_job_events
(
    id              INTEGER     not null
        primary key auto_increment,
    created         DATETIME    not null,
    status          VARCHAR(16) not null,
    progress        FLOAT       not null,
    comment         TEXT,
    finetune_job_id VARCHAR(36)
        references finetune_jobs
);

create table finetune_job_image_urls
(
    id              INTEGER       not null
        primary key auto_increment,
    url             VARCHAR(2048) not null,
    finetune_job_id VARCHAR(36)
        references finetune_jobs
);

create table inference_jobs
(
    id                    VARCHAR(36) not null
        primary key,
    prompt                TEXT,
    negative_prompt       TEXT,
    num_inference_steps   INTEGER     not null,
    num_images_per_prompt INTEGER     not null,
    guidance_scale        FLOAT       not null,
    aws_instance_id       VARCHAR(32)
        references aws_instances,
    finetune_job_id       VARCHAR(36)
        references finetune_jobs
);

create table generated_images
(
    url              VARCHAR(2048) not null,
    id               INTEGER       not null
        primary key auto_increment,
    inference_job_id VARCHAR(36)
        references inference_jobs
);

create table inference_job_events
(
    id               INTEGER     not null
        primary key auto_increment,
    created          DATETIME    not null,
    status           VARCHAR(16) not null,
    progress         FLOAT       not null,
    comment          TEXT,
    inference_job_id VARCHAR(36)
        references inference_jobs
);

