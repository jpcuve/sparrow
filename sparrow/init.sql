create table users
(
    id       int not null,
    username varchar(255),
    api_key  varchar(255),
    primary key (id)
);

create table training_requests
(
    id         varchar(64) not null,
    user_id    int         not null,
    parameters text,
    completed  int         not null default 0,
    primary key (id),
    foreign key (user_id) references users (id)
);

create table inference_requests
(
    id int not null,
);

insert into users
values (1, 'jp', 'jp-api-key'),
       (2, 'vicky', 'vicky-api-key');
