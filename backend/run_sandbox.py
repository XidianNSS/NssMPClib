import docker

def start_container(image_name: str, container_name: str = None, ports: dict = None, volumes: dict = None):
    """
    启动一个 Docker 容器

    :param image_name: 镜像名称 (如 'nginx:latest')
    :param container_name: 容器名称 (可选)
    :param ports: 端口映射 (如 {'80/tcp': 8080})
    :param volumes: 卷挂载 (如 {'/host/path': {'bind': '/container/path', 'mode': 'rw'}})
    :return: 容器对象
    """
    client = docker.from_env()

    try:
        container = client.containers.run(
            image=image_name,
            name=container_name,
            ports=ports,
            volumes=volumes,
            detach=True  # 在后台运行
        )
        print(f"容器 {container.short_id} 已启动")
        return container
    except Exception as e:
        print(f"启动容器失败: {e}")
        return None

# 示例用法
if __name__ == "__main__":
    # 启动一个nginx容器，映射主机8080到容器的80端口
    sandbox = start_container(
        image_name="sandbox:v2",
        container_name="sandbox_test",
        ports={'5001/tcp': 5001, '5002/tcp': 5002, '5003/tcp': 5003}
    )
