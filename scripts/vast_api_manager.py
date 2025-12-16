#!/usr/bin/env python3
"""
Vast.ai API Manager - GPU 实例自动化管理

基于 Vast.ai API 文档: https://docs.vast.ai/api-reference/

API 概览:
=========

1. 认证方式
-----------
- Bearer Token: Authorization: Bearer <api_key>
- API Key 可在 https://cloud.vast.ai/account/ 获取

2. 权限类别 (11种)
------------------
| 权限           | 用途                                |
|----------------|-------------------------------------|
| instance_read  | 获取实例信息和日志                  |
| instance_write | 创建/销毁/重启实例，执行命令        |
| user_read      | 读取用户数据                        |
| user_write     | 创建子账户，重置 API Key            |
| billing_read   | 查看账单                            |
| billing_write  | 转账                                |
| machine_read   | 查看可用机器                        |
| machine_write  | 管理机器上架/下架                   |
| misc           | 搜索 GPU offers，数据复制           |
| team_read      | 查看团队                            |
| team_write     | 管理团队                            |

3. 核心端点
-----------
- POST /api/v0/search/offers       搜索 GPU offers
- PUT  /api/v0/asks/{id}/          创建实例 (接受 offer)
- GET  /api/v0/instances/          列出所有实例
- GET  /api/v0/instances/{id}/     查看单个实例
- PUT  /api/v0/instances/{id}/     管理实例 (启动/停止)
- DELETE /api/v0/instances/{id}/   销毁实例
- PUT  /api/v0/instances/{id}/reboot/   重启
- PUT  /api/v0/instances/{id}/execute/  执行命令
- POST /api/v0/instances/{id}/ssh/      添加 SSH Key

4. 搜索过滤器运算符
-------------------
| 运算符 | 用途       | 示例                          |
|--------|------------|-------------------------------|
| eq     | 等于       | {"eq": true}                  |
| neq    | 不等于     | {"neq": false}                |
| gt     | 大于       | {"gt": 0.99}                  |
| lt     | 小于       | {"lt": 10000}                 |
| gte    | 大于等于   | {"gte": 4}                    |
| lte    | 小于等于   | {"lte": 8}                    |
| in     | 在列表中   | {"in": ["RTX_4090", "A100"]}  |
| nin    | 不在列表中 | {"nin": ["TW", "SE"]}         |

5. 创建实例请求体
-----------------
{
    "image": "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel",
    "disk": 100,                    # GB
    "price": 1.5,                   # $/hour (0.001-128)
    "runtype": "ssh",               # ssh|jupyter|args|ssh_proxy|ssh_direct
    "onstart": "pip install ...",   # 启动命令
    "env": {"KEY": "value"},        # 环境变量
    "label": "my-instance",         # 自定义名称
    "target_state": "running"       # running|stopped
}

6. 响应格式
-----------
创建成功: {"success": true, "new_contract": 1234568}
错误码: 400 (Bad Request), 401 (Unauthorized), 403 (Forbidden),
       404 (Not Found), 410 (Gone), 429 (Rate Limited)
"""

import os
import time
import json
import logging
import subprocess
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

@dataclass
class VastConfig:
    """Vast.ai 配置"""
    api_key: str = ""
    base_url: str = "https://console.vast.ai/api/v0"
    default_image: str = "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel"
    default_disk_gb: int = 100
    ssh_key_path: str = "~/.ssh/id_rsa.pub"

    # GPU 筛选默认条件
    min_gpu_ram_gb: int = 24
    min_reliability: float = 0.95
    preferred_gpus: List[str] = field(default_factory=lambda: [
        "RTX_4090", "RTX_A6000", "A100_PCIE", "A100_SXM4", "H100_PCIE", "H100_SXM5"
    ])

    # 训练相关
    project_path: str = "/Users/guboyang/Desktop/Project/FinSage"
    remote_path: str = "/root/FinSage"

    def __post_init__(self):
        # 从环境变量获取 API Key
        if not self.api_key:
            self.api_key = os.environ.get("VAST_API_KEY", "")


# ============================================================
# Vast.ai API Client
# ============================================================

class VastAPIClient:
    """
    Vast.ai API 客户端

    使用方法:
        client = VastAPIClient(api_key="your_key")

        # 搜索 GPU
        offers = client.search_offers(min_gpu_ram=48, gpu_names=["A100"])

        # 创建实例
        instance_id = client.create_instance(
            offer_id=offers[0]["id"],
            image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel",
            disk_gb=100,
            onstart="pip install transformers accelerate"
        )

        # 销毁实例
        client.destroy_instance(instance_id)
    """

    def __init__(self, api_key: str = None, config: VastConfig = None):
        if not HAS_REQUESTS:
            raise ImportError("requests library required. Run: pip install requests")

        self.config = config or VastConfig()
        self.api_key = api_key or self.config.api_key

        if not self.api_key:
            raise ValueError("API key required. Set VAST_API_KEY env or pass api_key")

        self.base_url = self.config.base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    # -------------------- Offers --------------------

    def search_offers(
        self,
        min_gpu_ram: int = 24,
        max_price: float = 5.0,
        gpu_names: List[str] = None,
        min_reliability: float = 0.95,
        num_gpus: int = 1,
        order_by: str = "dph_total",
        limit: int = 20,
    ) -> List[Dict]:
        """
        搜索可用 GPU offers

        Args:
            min_gpu_ram: 最小 GPU 显存 (GB)
            max_price: 最高价格 ($/hour)
            gpu_names: GPU 型号列表 (如 ["RTX_4090", "A100"])
            min_reliability: 最低可靠性
            num_gpus: GPU 数量
            order_by: 排序字段
            limit: 返回数量

        Returns:
            offers 列表
        """
        filters = {
            "gpu_ram": {"gte": min_gpu_ram * 1024},  # MB
            "dph_total": {"lte": max_price},
            "reliability": {"gte": min_reliability},
            "num_gpus": {"gte": num_gpus},
            "rentable": {"eq": True},
        }

        if gpu_names:
            filters["gpu_name"] = {"in": gpu_names}

        try:
            # 使用 GET /bundles 端点，将 filters 转为 JSON 字符串作为 query param
            import json as json_module
            resp = self.session.get(
                f"{self.base_url}/bundles",
                params={"q": json_module.dumps(filters), "limit": limit}
            )
            resp.raise_for_status()
            data = resp.json()

            offers = data.get("offers", [])

            # 排序
            if order_by == "dph_total":
                offers.sort(key=lambda x: x.get("dph_total", 999))
            elif order_by == "gpu_ram":
                offers.sort(key=lambda x: -x.get("gpu_ram", 0))

            return offers[:limit]

        except Exception as e:
            logger.error(f"Search offers failed: {e}")
            return []

    def get_best_offer(
        self,
        min_gpu_ram: int = 48,
        gpu_names: List[str] = None,
        max_price: float = 3.0,
    ) -> Optional[Dict]:
        """获取最佳 offer (价格最低)"""
        offers = self.search_offers(
            min_gpu_ram=min_gpu_ram,
            gpu_names=gpu_names or self.config.preferred_gpus,
            max_price=max_price,
            limit=1,
        )
        return offers[0] if offers else None

    # -------------------- Instances --------------------

    def create_instance(
        self,
        offer_id: int,
        image: str = None,
        disk_gb: int = 100,
        onstart: str = "",
        env: Dict[str, str] = None,
        label: str = None,
        price: float = None,
    ) -> Optional[int]:
        """
        创建实例

        Args:
            offer_id: Offer ID (ask_id)
            image: Docker 镜像
            disk_gb: 磁盘大小
            onstart: 启动命令
            env: 环境变量
            label: 实例标签
            price: 出价 ($/hour)

        Returns:
            instance_id (contract_id)
        """
        body = {
            "image": image or self.config.default_image,
            "disk": disk_gb,
            "runtype": "ssh",
        }

        if onstart:
            body["onstart"] = onstart
        if env:
            body["env"] = env
        if label:
            body["label"] = label
        if price:
            body["price"] = price

        try:
            resp = self.session.put(
                f"{self.base_url}/asks/{offer_id}/",
                json=body
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("success"):
                instance_id = data.get("new_contract")
                logger.info(f"Instance created: {instance_id}")
                return instance_id
            else:
                logger.error(f"Create instance failed: {data}")
                return None

        except Exception as e:
            logger.error(f"Create instance failed: {e}")
            return None

    def list_instances(self) -> List[Dict]:
        """列出所有实例"""
        try:
            resp = self.session.get(f"{self.base_url}/instances/")
            resp.raise_for_status()
            return resp.json().get("instances", [])
        except Exception as e:
            logger.error(f"List instances failed: {e}")
            return []

    def get_instance(self, instance_id: int) -> Optional[Dict]:
        """获取实例详情"""
        try:
            resp = self.session.get(f"{self.base_url}/instances/{instance_id}/")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Get instance failed: {e}")
            return None

    def destroy_instance(self, instance_id: int) -> bool:
        """销毁实例"""
        try:
            resp = self.session.delete(f"{self.base_url}/instances/{instance_id}/")
            resp.raise_for_status()
            logger.info(f"Instance {instance_id} destroyed")
            return True
        except Exception as e:
            logger.error(f"Destroy instance failed: {e}")
            return False

    def start_instance(self, instance_id: int) -> bool:
        """启动实例"""
        return self._manage_instance(instance_id, "running")

    def stop_instance(self, instance_id: int) -> bool:
        """停止实例"""
        return self._manage_instance(instance_id, "stopped")

    def _manage_instance(self, instance_id: int, target_state: str) -> bool:
        """管理实例状态"""
        try:
            resp = self.session.put(
                f"{self.base_url}/instances/{instance_id}/",
                json={"target_state": target_state}
            )
            resp.raise_for_status()
            logger.info(f"Instance {instance_id} -> {target_state}")
            return True
        except Exception as e:
            logger.error(f"Manage instance failed: {e}")
            return False

    def reboot_instance(self, instance_id: int) -> bool:
        """重启实例"""
        try:
            resp = self.session.put(f"{self.base_url}/instances/{instance_id}/reboot/")
            resp.raise_for_status()
            logger.info(f"Instance {instance_id} rebooting")
            return True
        except Exception as e:
            logger.error(f"Reboot instance failed: {e}")
            return False

    def attach_ssh_key(self, instance_id: int, ssh_key: str) -> bool:
        """添加 SSH Key"""
        try:
            resp = self.session.post(
                f"{self.base_url}/instances/{instance_id}/ssh/",
                json={"ssh_key": ssh_key}
            )
            resp.raise_for_status()
            logger.info(f"SSH key attached to instance {instance_id}")
            return True
        except Exception as e:
            logger.error(f"Attach SSH key failed: {e}")
            return False

    def get_ssh_info(self, instance_id: int) -> Optional[Dict]:
        """获取 SSH 连接信息"""
        instance = self.get_instance(instance_id)
        if not instance:
            return None

        return {
            "host": instance.get("public_ipaddr") or instance.get("ssh_host"),
            "port": instance.get("ssh_port", 22),
            "user": "root",
            "command": f"ssh -p {instance.get('ssh_port', 22)} root@{instance.get('ssh_host', instance.get('public_ipaddr'))}"
        }

    # -------------------- Wait Helpers --------------------

    def wait_for_instance(
        self,
        instance_id: int,
        target_status: str = "running",
        timeout: int = 600,
        poll_interval: int = 10,
    ) -> bool:
        """
        等待实例达到目标状态

        Args:
            instance_id: 实例 ID
            target_status: 目标状态 ("running", "stopped", etc.)
            timeout: 超时秒数
            poll_interval: 轮询间隔

        Returns:
            是否成功
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            instance = self.get_instance(instance_id)
            if not instance:
                logger.warning(f"Instance {instance_id} not found")
                return False

            current_status = instance.get("actual_status", "")
            logger.debug(f"Instance {instance_id} status: {current_status}")

            if current_status == target_status:
                return True

            time.sleep(poll_interval)

        logger.error(f"Timeout waiting for instance {instance_id} -> {target_status}")
        return False


# ============================================================
# Training Job Manager
# ============================================================

class VastTrainingManager:
    """
    训练任务管理器

    自动化流程:
    1. 搜索最佳 GPU offer
    2. 创建实例
    3. 等待实例就绪
    4. 同步代码
    5. 启动训练
    6. 监控进度
    7. 训练完成后销毁实例
    """

    def __init__(self, api_key: str = None, config: VastConfig = None):
        self.config = config or VastConfig()
        self.client = VastAPIClient(api_key=api_key, config=self.config)
        self.current_instance_id: Optional[int] = None

    def provision_gpu(
        self,
        min_gpu_ram: int = 48,
        gpu_names: List[str] = None,
        max_price: float = 3.0,
        onstart: str = None,
    ) -> Optional[Dict]:
        """
        自动配置 GPU 实例

        Returns:
            实例信息 (包含 SSH 连接信息)
        """
        # 1. 搜索最佳 offer
        logger.info(f"Searching for GPU with >= {min_gpu_ram}GB RAM...")
        offer = self.client.get_best_offer(
            min_gpu_ram=min_gpu_ram,
            gpu_names=gpu_names,
            max_price=max_price,
        )

        if not offer:
            logger.error("No suitable GPU offer found")
            return None

        logger.info(f"Found offer: {offer.get('gpu_name')} "
                   f"({offer.get('gpu_ram', 0) // 1024}GB) "
                   f"@ ${offer.get('dph_total', 0):.2f}/hr")

        # 2. 默认启动命令
        if onstart is None:
            onstart = """
pip install -q transformers accelerate peft bitsandbytes
pip install -q tenacity python-dotenv requests pandas numpy yfinance
echo 'Dependencies installed'
"""

        # 3. 创建实例
        instance_id = self.client.create_instance(
            offer_id=offer["id"],
            disk_gb=self.config.default_disk_gb,
            onstart=onstart,
            label=f"finsage-training-{datetime.now().strftime('%Y%m%d-%H%M')}",
        )

        if not instance_id:
            return None

        self.current_instance_id = instance_id

        # 4. 等待实例就绪
        logger.info(f"Waiting for instance {instance_id} to be ready...")
        if not self.client.wait_for_instance(instance_id, timeout=600):
            logger.error("Instance failed to start")
            self.client.destroy_instance(instance_id)
            return None

        # 5. 获取 SSH 信息
        ssh_info = self.client.get_ssh_info(instance_id)

        return {
            "instance_id": instance_id,
            "gpu_name": offer.get("gpu_name"),
            "gpu_ram_gb": offer.get("gpu_ram", 0) // 1024,
            "price_per_hour": offer.get("dph_total", 0),
            "ssh_info": ssh_info,
        }

    def sync_code(self, instance_id: int = None) -> bool:
        """同步代码到远程实例"""
        instance_id = instance_id or self.current_instance_id
        if not instance_id:
            logger.error("No instance ID")
            return False

        ssh_info = self.client.get_ssh_info(instance_id)
        if not ssh_info:
            return False

        # rsync 同步
        cmd = [
            "rsync", "-avz", "--delete",
            "-e", f"ssh -p {ssh_info['port']} -o StrictHostKeyChecking=no",
            f"{self.config.project_path}/",
            f"root@{ssh_info['host']}:{self.config.remote_path}/"
        ]

        try:
            logger.info(f"Syncing code to {ssh_info['host']}:{ssh_info['port']}...")
            subprocess.run(cmd, check=True)
            logger.info("Code sync completed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Code sync failed: {e}")
            return False

    def run_training(
        self,
        instance_id: int = None,
        script: str = "scripts/train_with_real_data_v3.py",
        args: str = "",
        background: bool = True,
    ) -> bool:
        """在远程实例上运行训练"""
        instance_id = instance_id or self.current_instance_id
        if not instance_id:
            logger.error("No instance ID")
            return False

        ssh_info = self.client.get_ssh_info(instance_id)
        if not ssh_info:
            return False

        # 构建训练命令
        train_cmd = f"cd {self.config.remote_path} && python3 {script} {args}"

        if background:
            train_cmd = f"nohup {train_cmd} > /root/training.log 2>&1 &"

        ssh_cmd = [
            "ssh",
            "-p", str(ssh_info["port"]),
            "-o", "StrictHostKeyChecking=no",
            f"root@{ssh_info['host']}",
            train_cmd
        ]

        try:
            logger.info(f"Starting training: {script}")
            subprocess.run(ssh_cmd, check=True)
            logger.info("Training started")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Training start failed: {e}")
            return False

    def cleanup(self, instance_id: int = None) -> bool:
        """销毁实例"""
        instance_id = instance_id or self.current_instance_id
        if instance_id:
            return self.client.destroy_instance(instance_id)
        return False


# ============================================================
# CLI Interface
# ============================================================

def main():
    """命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description="Vast.ai GPU Manager")
    subparsers = parser.add_subparsers(dest="command")

    # search 命令
    search_parser = subparsers.add_parser("search", help="Search GPU offers")
    search_parser.add_argument("--gpu-ram", type=int, default=24, help="Min GPU RAM (GB)")
    search_parser.add_argument("--max-price", type=float, default=5.0, help="Max price ($/hr)")
    search_parser.add_argument("--gpu", type=str, nargs="+", help="GPU names")
    search_parser.add_argument("--limit", type=int, default=10, help="Number of results")

    # create 命令
    create_parser = subparsers.add_parser("create", help="Create instance")
    create_parser.add_argument("--offer-id", type=int, required=True, help="Offer ID")
    create_parser.add_argument("--disk", type=int, default=100, help="Disk size (GB)")
    create_parser.add_argument("--image", type=str, help="Docker image")

    # list 命令
    subparsers.add_parser("list", help="List instances")

    # destroy 命令
    destroy_parser = subparsers.add_parser("destroy", help="Destroy instance")
    destroy_parser.add_argument("instance_id", type=int, help="Instance ID")

    # provision 命令 (自动配置)
    provision_parser = subparsers.add_parser("provision", help="Auto provision GPU")
    provision_parser.add_argument("--gpu-ram", type=int, default=48, help="Min GPU RAM (GB)")
    provision_parser.add_argument("--max-price", type=float, default=3.0, help="Max price ($/hr)")

    args = parser.parse_args()

    # 初始化客户端
    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        print("Error: Set VAST_API_KEY environment variable")
        return

    client = VastAPIClient(api_key=api_key)

    if args.command == "search":
        offers = client.search_offers(
            min_gpu_ram=args.gpu_ram,
            max_price=args.max_price,
            gpu_names=args.gpu,
            limit=args.limit,
        )
        print(f"\nFound {len(offers)} offers:\n")
        for i, offer in enumerate(offers):
            print(f"{i+1}. ID: {offer['id']}")
            print(f"   GPU: {offer.get('gpu_name')} ({offer.get('gpu_ram', 0) // 1024}GB)")
            print(f"   Price: ${offer.get('dph_total', 0):.3f}/hr")
            print(f"   Reliability: {offer.get('reliability', 0):.1%}")
            print()

    elif args.command == "create":
        instance_id = client.create_instance(
            offer_id=args.offer_id,
            disk_gb=args.disk,
            image=args.image,
        )
        if instance_id:
            print(f"Instance created: {instance_id}")
        else:
            print("Failed to create instance")

    elif args.command == "list":
        instances = client.list_instances()
        print(f"\n{len(instances)} instances:\n")
        for inst in instances:
            print(f"ID: {inst.get('id')}")
            print(f"   Status: {inst.get('actual_status')}")
            print(f"   GPU: {inst.get('gpu_name')}")
            print(f"   SSH: ssh -p {inst.get('ssh_port')} root@{inst.get('ssh_host')}")
            print()

    elif args.command == "destroy":
        if client.destroy_instance(args.instance_id):
            print(f"Instance {args.instance_id} destroyed")
        else:
            print("Failed to destroy instance")

    elif args.command == "provision":
        manager = VastTrainingManager(api_key=api_key)
        result = manager.provision_gpu(
            min_gpu_ram=args.gpu_ram,
            max_price=args.max_price,
        )
        if result:
            print(f"\nGPU Provisioned:")
            print(f"  Instance ID: {result['instance_id']}")
            print(f"  GPU: {result['gpu_name']} ({result['gpu_ram_gb']}GB)")
            print(f"  Price: ${result['price_per_hour']:.3f}/hr")
            print(f"  SSH: {result['ssh_info']['command']}")
        else:
            print("Failed to provision GPU")

    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
