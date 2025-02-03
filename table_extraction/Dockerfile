FROM ankane/pgvector

RUN echo 'DO $$ \
BEGIN \
    IF NOT EXISTS (SELECT FROM pg_tables WHERE schemaname = '\''public'\'' AND tablename = '\''page'\'') THEN \
        CREATE TABLE Page ( \
            id VARCHAR(255) PRIMARY KEY, \
            title VARCHAR(255) NOT NULL, \
            source VARCHAR(255) NOT NULL, \
            page_content TEXT NOT NULL, \
            page_nbr INTEGER NOT NULL, \
            model_name VARCHAR(255) NOT NULL, \
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP \
        ); \
    END IF; \
END $$;' > /docker-entrypoint-initdb.d/create_page_table.sql

CMD ["sh", "-c", "docker-entrypoint.sh postgres & sleep 10 && psql -U \"$POSTGRES_USER\" -d \"$POSTGRES_DB\" -f /docker-entrypoint-initdb.d/create_page_table.sql && wait"]
