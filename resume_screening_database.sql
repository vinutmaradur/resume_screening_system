CREATE DATABASE resume_database;

USE resume_database;

CREATE TABLE resume (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255),
    mobile_number VARCHAR(20),
    no_of_pages INT,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Predicted_Field VARCHAR(255),
    User_level VARCHAR(255),
    skills TEXT,
    recommended_skills TEXT,
    courses TEXT
);

DESCRIBE resume;

GRANT ALL PRIVILEGES ON resume_database.* TO 'root'@'localhost';
FLUSH PRIVILEGES;

SELECT * FROM resume;
