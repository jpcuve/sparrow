FROM python:3.11-slim-bullseye

# setup dependencies
RUN apt-get update
RUN apt-get install xz-utils
RUN apt-get -y install curl

# Download latest nodejs binary
ENV NODE_VERSION=v19.3.0
RUN curl https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-x64.tar.xz -O

# Extract & install
RUN tar -xf node-${NODE_VERSION}-linux-x64.tar.xz
RUN ln -s /node-${NODE_VERSION}-linux-x64/bin/node /usr/local/bin/node
RUN ln -s /node-${NODE_VERSION}-linux-x64/bin/npm /usr/local/bin/npm
RUN ln -s /node-${NODE_VERSION}-linux-x64/bin/npx /usr/local/bin/npx

# Front end
COPY ./app/ /app/app/
WORKDIR /app/app
RUN npm install
RUN npm run build

# Back end
COPY ./requirements.txt /app/requirements.txt
COPY ./sparrow/ /app/sparrow/
COPY ./*.py /app/
WORKDIR /app
RUN pip install -r requirements.txt
RUN ls -la
EXPOSE 5000
ENTRYPOINT gunicorn -b 0.0.0.0:5000 -w 2 main:app