import json
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)


# ============================================================
# 2. MOCK DEVICE KPI
#
# Daily aggregated health data.
# ============================================================

MOCK_DEVICE_KPI = [

    # ========================================================
    # ROUTER-001
    #
    # Scenario:
    # Broad degradation after firmware/configuration activity.
    # ========================================================

    {
        "device_id": "ROUTER-001",
        "kpi_date": "2026-08-12",
        "wifi_score": 92,
        "signal_strength_quality": 90,
        "avg_signal_strength": -70,
        "min_signal_strength": -74,
        "max_signal_strength": -66,
        "uptime_percentage": 99.8,
        "avg_download_speed": 94.5,
        "avg_upload_speed": 48.2,
        "avg_latency_ms": 12,
        "packet_loss_percentage": 0.2,
        "overall_health_score": 94,
    },
    {
        "device_id": "ROUTER-001",
        "kpi_date": "2026-08-13",
        "wifi_score": 87,
        "signal_strength_quality": 84,
        "avg_signal_strength": -75,
        "min_signal_strength": -80,
        "max_signal_strength": -70,
        "uptime_percentage": 99.1,
        "avg_download_speed": 86.3,
        "avg_upload_speed": 43.7,
        "avg_latency_ms": 18,
        "packet_loss_percentage": 0.8,
        "overall_health_score": 86,
    },
    {
        "device_id": "ROUTER-001",
        "kpi_date": "2026-08-14",
        "wifi_score": 51,
        "signal_strength_quality": 45,
        "avg_signal_strength": -83.5,
        "min_signal_strength": -91,
        "max_signal_strength": -72,
        "uptime_percentage": 91.4,
        "avg_download_speed": 42.1,
        "avg_upload_speed": 18.4,
        "avg_latency_ms": 57,
        "packet_loss_percentage": 6.8,
        "overall_health_score": 48,
    },

    # ========================================================
    # ROUTER-002
    #
    # Scenario:
    # WiFi/signal/uptime healthy.
    # WAN/performance is degraded.
    # ========================================================

    {
        "device_id": "ROUTER-002",
        "kpi_date": "2026-08-12",
        "wifi_score": 94,
        "signal_strength_quality": 93,
        "avg_signal_strength": -68,
        "min_signal_strength": -72,
        "max_signal_strength": -63,
        "uptime_percentage": 99.9,
        "avg_download_speed": 95.0,
        "avg_upload_speed": 49.0,
        "avg_latency_ms": 11,
        "packet_loss_percentage": 0.1,
        "overall_health_score": 95,
    },
    {
        "device_id": "ROUTER-002",
        "kpi_date": "2026-08-13",
        "wifi_score": 93,
        "signal_strength_quality": 92,
        "avg_signal_strength": -69,
        "min_signal_strength": -73,
        "max_signal_strength": -64,
        "uptime_percentage": 99.8,
        "avg_download_speed": 48.0,
        "avg_upload_speed": 21.0,
        "avg_latency_ms": 62,
        "packet_loss_percentage": 5.2,
        "overall_health_score": 66,
    },
    {
        "device_id": "ROUTER-002",
        "kpi_date": "2026-08-14",
        "wifi_score": 94,
        "signal_strength_quality": 93,
        "avg_signal_strength": -68,
        "min_signal_strength": -72,
        "max_signal_strength": -63,
        "uptime_percentage": 99.7,
        "avg_download_speed": 45.2,
        "avg_upload_speed": 19.8,
        "avg_latency_ms": 68,
        "packet_loss_percentage": 5.9,
        "overall_health_score": 64,
    },

    # ========================================================
    # ROUTER-003
    #
    # Scenario:
    # WiFi and performance healthy.
    # Device stability / uptime is poor.
    # ========================================================

    {
        "device_id": "ROUTER-003",
        "kpi_date": "2026-08-12",
        "wifi_score": 91,
        "signal_strength_quality": 89,
        "avg_signal_strength": -71,
        "min_signal_strength": -76,
        "max_signal_strength": -67,
        "uptime_percentage": 99.7,
        "avg_download_speed": 91.2,
        "avg_upload_speed": 46.8,
        "avg_latency_ms": 14,
        "packet_loss_percentage": 0.3,
        "overall_health_score": 92,
    },
    {
        "device_id": "ROUTER-003",
        "kpi_date": "2026-08-13",
        "wifi_score": 90,
        "signal_strength_quality": 88,
        "avg_signal_strength": -72,
        "min_signal_strength": -77,
        "max_signal_strength": -68,
        "uptime_percentage": 84.3,
        "avg_download_speed": 89.7,
        "avg_upload_speed": 45.9,
        "avg_latency_ms": 15,
        "packet_loss_percentage": 0.4,
        "overall_health_score": 75,
    },
    {
        "device_id": "ROUTER-003",
        "kpi_date": "2026-08-14",
        "wifi_score": 90,
        "signal_strength_quality": 88,
        "avg_signal_strength": -72,
        "min_signal_strength": -77,
        "max_signal_strength": -68,
        "uptime_percentage": 78.2,
        "avg_download_speed": 90.1,
        "avg_upload_speed": 46.1,
        "avg_latency_ms": 15,
        "packet_loss_percentage": 0.4,
        "overall_health_score": 72,
    },
]


# ============================================================
# 3. MOCK RAW DEVICE METRIC
#
# In production this represents data persisted from Kafka
# into device_metric.
# ============================================================

MOCK_DEVICE_METRIC = [

    # ========================================================
    # ROUTER-001
    # ========================================================

    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T07:30:00+08:00",
        "metric_name": "wifi_score",
        "metric_value": 86,
        "unit": "score",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T07:30:00+08:00",
        "metric_name": "signal_strength",
        "metric_value": -72,
        "unit": "dBm",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T07:30:00+08:00",
        "metric_name": "download_speed",
        "metric_value": 88,
        "unit": "Mbps",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T07:30:00+08:00",
        "metric_name": "upload_speed",
        "metric_value": 44,
        "unit": "Mbps",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T07:30:00+08:00",
        "metric_name": "latency",
        "metric_value": 18,
        "unit": "ms",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T07:30:00+08:00",
        "metric_name": "packet_loss",
        "metric_value": 0.4,
        "unit": "%",
    },

    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T08:00:00+08:00",
        "metric_name": "wifi_score",
        "metric_value": 72,
        "unit": "score",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T08:00:00+08:00",
        "metric_name": "signal_strength",
        "metric_value": -80,
        "unit": "dBm",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T08:00:00+08:00",
        "metric_name": "download_speed",
        "metric_value": 65,
        "unit": "Mbps",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T08:00:00+08:00",
        "metric_name": "upload_speed",
        "metric_value": 31,
        "unit": "Mbps",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T08:00:00+08:00",
        "metric_name": "latency",
        "metric_value": 31,
        "unit": "ms",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T08:00:00+08:00",
        "metric_name": "packet_loss",
        "metric_value": 2.0,
        "unit": "%",
    },

    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T08:30:00+08:00",
        "metric_name": "wifi_score",
        "metric_value": 59,
        "unit": "score",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T08:30:00+08:00",
        "metric_name": "signal_strength",
        "metric_value": -86,
        "unit": "dBm",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T08:30:00+08:00",
        "metric_name": "download_speed",
        "metric_value": 45,
        "unit": "Mbps",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T08:30:00+08:00",
        "metric_name": "upload_speed",
        "metric_value": 20,
        "unit": "Mbps",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T08:30:00+08:00",
        "metric_name": "latency",
        "metric_value": 55,
        "unit": "ms",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T08:30:00+08:00",
        "metric_name": "packet_loss",
        "metric_value": 5.5,
        "unit": "%",
    },

    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T09:00:00+08:00",
        "metric_name": "wifi_score",
        "metric_value": 48,
        "unit": "score",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T09:00:00+08:00",
        "metric_name": "signal_strength",
        "metric_value": -91,
        "unit": "dBm",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T09:00:00+08:00",
        "metric_name": "download_speed",
        "metric_value": 39,
        "unit": "Mbps",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T09:00:00+08:00",
        "metric_name": "upload_speed",
        "metric_value": 17,
        "unit": "Mbps",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T09:00:00+08:00",
        "metric_name": "latency",
        "metric_value": 72,
        "unit": "ms",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T09:00:00+08:00",
        "metric_name": "packet_loss",
        "metric_value": 8.0,
        "unit": "%",
    },

    # ========================================================
    # ROUTER-002
    #
    # Performance degradation without WiFi degradation.
    # ========================================================

    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-14T06:00:00+08:00",
        "metric_name": "wifi_score",
        "metric_value": 94,
        "unit": "score",
    },
    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-14T06:00:00+08:00",
        "metric_name": "signal_strength",
        "metric_value": -68,
        "unit": "dBm",
    },
    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-14T06:00:00+08:00",
        "metric_name": "download_speed",
        "metric_value": 91,
        "unit": "Mbps",
    },
    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-14T06:00:00+08:00",
        "metric_name": "upload_speed",
        "metric_value": 47,
        "unit": "Mbps",
    },
    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-14T06:00:00+08:00",
        "metric_name": "latency",
        "metric_value": 13,
        "unit": "ms",
    },
    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-14T06:00:00+08:00",
        "metric_name": "packet_loss",
        "metric_value": 0.2,
        "unit": "%",
    },

    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-14T08:00:00+08:00",
        "metric_name": "wifi_score",
        "metric_value": 94,
        "unit": "score",
    },
    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-14T08:00:00+08:00",
        "metric_name": "signal_strength",
        "metric_value": -69,
        "unit": "dBm",
    },
    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-14T08:00:00+08:00",
        "metric_name": "download_speed",
        "metric_value": 48,
        "unit": "Mbps",
    },
    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-14T08:00:00+08:00",
        "metric_name": "upload_speed",
        "metric_value": 21,
        "unit": "Mbps",
    },
    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-14T08:00:00+08:00",
        "metric_name": "latency",
        "metric_value": 62,
        "unit": "ms",
    },
    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-14T08:00:00+08:00",
        "metric_name": "packet_loss",
        "metric_value": 5.2,
        "unit": "%",
    },

    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-14T10:00:00+08:00",
        "metric_name": "wifi_score",
        "metric_value": 94,
        "unit": "score",
    },
    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-14T10:00:00+08:00",
        "metric_name": "signal_strength",
        "metric_value": -68,
        "unit": "dBm",
    },
    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-14T10:00:00+08:00",
        "metric_name": "download_speed",
        "metric_value": 45,
        "unit": "Mbps",
    },
    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-14T10:00:00+08:00",
        "metric_name": "upload_speed",
        "metric_value": 20,
        "unit": "Mbps",
    },
    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-14T10:00:00+08:00",
        "metric_name": "latency",
        "metric_value": 68,
        "unit": "ms",
    },
    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-14T10:00:00+08:00",
        "metric_name": "packet_loss",
        "metric_value": 5.9,
        "unit": "%",
    },

    # ========================================================
    # ROUTER-003
    #
    # Stable WiFi/performance, repeated uptime interruptions.
    # ========================================================

    {
        "device_id": "ROUTER-003",
        "timestamp": "2026-08-14T02:00:00+08:00",
        "metric_name": "wifi_score",
        "metric_value": 90,
        "unit": "score",
    },
    {
        "device_id": "ROUTER-003",
        "timestamp": "2026-08-14T02:00:00+08:00",
        "metric_name": "signal_strength",
        "metric_value": -72,
        "unit": "dBm",
    },
    {
        "device_id": "ROUTER-003",
        "timestamp": "2026-08-14T02:00:00+08:00",
        "metric_name": "download_speed",
        "metric_value": 90,
        "unit": "Mbps",
    },
    {
        "device_id": "ROUTER-003",
        "timestamp": "2026-08-14T02:00:00+08:00",
        "metric_name": "upload_speed",
        "metric_value": 46,
        "unit": "Mbps",
    },
    {
        "device_id": "ROUTER-003",
        "timestamp": "2026-08-14T02:00:00+08:00",
        "metric_name": "latency",
        "metric_value": 15,
        "unit": "ms",
    },
    {
        "device_id": "ROUTER-003",
        "timestamp": "2026-08-14T02:00:00+08:00",
        "metric_name": "packet_loss",
        "metric_value": 0.3,
        "unit": "%",
    },
]


# ============================================================
# 4. MOCK OPERATION LOG
# ============================================================

MOCK_OPERATION_LOG = [

    # ROUTER-001
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T07:40:00+08:00",
        "operation": "firmware_upgrade",
        "status": "success",
        "details": {
            "from_version": "1.2.1",
            "to_version": "1.3.0",
        },
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T07:50:00+08:00",
        "operation": "router_reboot",
        "status": "success",
        "details": {},
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T08:05:00+08:00",
        "operation": "wifi_configuration_change",
        "status": "success",
        "details": {
            "channel": "11",
        },
    },

    # ROUTER-002
    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-13T06:30:00+08:00",
        "operation": "wan_configuration_change",
        "status": "success",
        "details": {
            "connection_mode": "auto",
        },
    },

    # ROUTER-003
    {
        "device_id": "ROUTER-003",
        "timestamp": "2026-08-13T10:10:00+08:00",
        "operation": "router_reboot",
        "status": "success",
        "details": {},
    },
    {
        "device_id": "ROUTER-003",
        "timestamp": "2026-08-13T13:45:00+08:00",
        "operation": "router_reboot",
        "status": "success",
        "details": {},
    },
    {
        "device_id": "ROUTER-003",
        "timestamp": "2026-08-14T02:15:00+08:00",
        "operation": "router_reboot",
        "status": "success",
        "details": {},
    },
    {
        "device_id": "ROUTER-003",
        "timestamp": "2026-08-14T05:20:00+08:00",
        "operation": "router_reboot",
        "status": "success",
        "details": {},
    },
]


# ============================================================
# 5. MOCK SESSION HISTORY
# ============================================================

MOCK_SESSION_HISTORY = [

    # ROUTER-001
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T07:30:00+08:00",
        "user_id": "USER-1001",
        "action": "request_firmware_upgrade",
        "status": "completed",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T07:35:00+08:00",
        "user_id": "USER-1001",
        "action": "confirm_firmware_upgrade",
        "status": "completed",
    },
    {
        "device_id": "ROUTER-001",
        "timestamp": "2026-08-14T08:02:00+08:00",
        "user_id": "USER-1001",
        "action": "change_wifi_channel",
        "status": "completed",
    },

    # ROUTER-002
    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-13T06:20:00+08:00",
        "user_id": "USER-1002",
        "action": "change_wan_configuration",
        "status": "completed",
    },
    {
        "device_id": "ROUTER-002",
        "timestamp": "2026-08-13T07:00:00+08:00",
        "user_id": "USER-1002",
        "action": "run_speed_test",
        "status": "completed",
    },

    # ROUTER-003
    {
        "device_id": "ROUTER-003",
        "timestamp": "2026-08-13T10:00:00+08:00",
        "user_id": "USER-1003",
        "action": "request_router_restart",
        "status": "completed",
    },
    {
        "device_id": "ROUTER-003",
        "timestamp": "2026-08-13T13:30:00+08:00",
        "user_id": "USER-1003",
        "action": "request_router_restart",
        "status": "completed",
    },
]


# ============================================================
# 6. TOOLS
# ============================================================

@tool
def get_device_health_kpi(device_id: str, days: int = 3) -> str:
    """
    Retrieve the latest daily health KPI records for a router.

    Use this for a general device health assessment.

    Returns:
    - WiFi score
    - signal strength quality
    - average/min/max signal strength
    - uptime
    - download speed
    - upload speed
    - latency
    - packet loss
    - overall health score
    """

    print(
        f"\n[TOOL] get_device_health_kpi("
        f"device_id={device_id}, days={days})"
    )

    data = [
        item
        for item in MOCK_DEVICE_KPI
        if item["device_id"] == device_id
    ]

    data = sorted(
        data,
        key=lambda x: x["kpi_date"],
        reverse=True,
    )

    return json.dumps(data[:days], indent=2)


@tool
def get_device_kpi(device_id: str, days: int = 3) -> str:
    """
    Retrieve daily device KPI records.

    Use this when a historical KPI comparison is required,
    especially when identifying the worst day for a metric.
    """

    print(
        f"\n[TOOL] get_device_kpi("
        f"device_id={device_id}, days={days})"
    )

    data = [
        item
        for item in MOCK_DEVICE_KPI
        if item["device_id"] == device_id
    ]

    data = sorted(
        data,
        key=lambda x: x["kpi_date"],
        reverse=True,
    )

    return json.dumps(data[:days], indent=2)


@tool
def get_device_metrics(
    device_id: str,
    date: str,
) -> str:
    """
    Retrieve all raw device metrics for a router on a specific date.

    Use this after a health/KPI assessment identifies a
    problematic date or category.

    This represents the device_metric table populated from Kafka.
    """

    print(
        f"\n[TOOL] get_device_metrics("
        f"device_id={device_id}, date={date})"
    )

    data = [
        item
        for item in MOCK_DEVICE_METRIC
        if (
            item["device_id"] == device_id
            and item["timestamp"].startswith(date)
        )
    ]

    data = sorted(
        data,
        key=lambda x: x["timestamp"],
    )

    return json.dumps(data, indent=2)


@tool
def get_device_metric(
    device_id: str,
    date: str,
    metric_name: str,
) -> str:
    """
    Retrieve one raw metric for a router on a specific date.

    Examples:
    signal_strength
    wifi_score
    download_speed
    upload_speed
    latency
    packet_loss
    """

    print(
        f"\n[TOOL] get_device_metric("
        f"device_id={device_id}, "
        f"date={date}, "
        f"metric={metric_name})"
    )

    data = [
        item
        for item in MOCK_DEVICE_METRIC
        if (
            item["device_id"] == device_id
            and item["timestamp"].startswith(date)
            and item["metric_name"] == metric_name
        )
    ]

    data = sorted(
        data,
        key=lambda x: x["timestamp"],
    )

    return json.dumps(data, indent=2)


@tool
def get_operation_history(
    device_id: str,
    date: str,
) -> str:
    """
    Retrieve router operations performed on a specific date.
    """

    print(
        f"\n[TOOL] get_operation_history("
        f"device_id={device_id}, date={date})"
    )

    data = [
        item
        for item in MOCK_OPERATION_LOG
        if (
            item["device_id"] == device_id
            and item["timestamp"].startswith(date)
        )
    ]

    return json.dumps(data, indent=2)


@tool
def get_session_history(
    device_id: str,
    date: str,
) -> str:
    """
    Retrieve user actions performed on a specific date.
    """

    print(
        f"\n[TOOL] get_session_history("
        f"device_id={device_id}, date={date})"
    )

    data = [
        item
        for item in MOCK_SESSION_HISTORY
        if (
            item["device_id"] == device_id
            and item["timestamp"].startswith(date)
        )
    ]

    return json.dumps(data, indent=2)


@tool
def classify_signal_strength(signal_dbm: float) -> str:
    """
    Classify WiFi signal strength using a deterministic rule.

    - >= -70 dBm: GOOD
    - >= -80 dBm: MODERATE
    - < -80 dBm: WEAK
    """

    print(
        f"\n[TOOL] classify_signal_strength("
        f"{signal_dbm} dBm)"
    )

    if signal_dbm >= -70:
        return "GOOD"

    if signal_dbm >= -80:
        return "MODERATE"

    return "WEAK"


# ============================================================
# 7. HEALTH AGENT
# ============================================================

health_agent = create_react_agent(
    model=model,
    tools=[get_device_health_kpi],
    name="health_expert",
    prompt="""
You are a Router Health Assessment Specialist.

Your job is to perform the FIRST high-level assessment
when the user asks a general question such as:

- "How is my router?"
- "How is ROUTER-001?"
- "Is there anything wrong with my device?"
- "What's the health status?"

Use get_device_health_kpi.

Evaluate these categories:

1. WiFi quality
2. Signal strength
3. Device uptime/stability
4. Download speed
5. Upload speed
6. Latency
7. Packet loss
8. Overall health

Compare the latest available days and identify:

- healthy categories
- degraded categories
- improving/degrading trends
- the most important abnormality
- the date that needs deeper investigation

IMPORTANT:

Do not determine root cause.

Your job is to determine WHAT appears unhealthy,
not WHY it happened.

Clearly state the device_id and dates in your findings.

After producing the health assessment, stop.
Do not repeatedly call the same tool.
""",
)


# ============================================================
# 8. KPI AGENT
# ============================================================

kpi_agent = create_react_agent(
    model=model,
    tools=[get_device_kpi],
    name="kpi_expert",
    prompt="""
You are a Device KPI Specialist.

Use this specialist for historical KPI analysis when
the supervisor needs a comparison across multiple days.

Use get_device_kpi.

Determine:

- latest 3 days
- relevant KPI values
- worst day
- strongest/weakest trend
- date that requires raw metric investigation

Do not investigate raw metrics.
Do not investigate operation logs.
Do not investigate user sessions.
Do not claim root cause.

Return concise findings with exact values.

After producing the KPI analysis, stop.
""",
)


# ============================================================
# 9. METRIC AGENT
# ============================================================

metric_agent = create_react_agent(
    model=model,
    tools=[
        get_device_metrics,
        get_device_metric,
        classify_signal_strength,
    ],
    name="metric_expert",
    prompt="""
You are a Device Metric Specialist.

Your job is to drill down into raw device_metric data
after a problematic category/date has been identified.

Use get_device_metrics to inspect the raw measurements.

The available metrics include:

- wifi_score
- signal_strength
- download_speed
- upload_speed
- latency
- packet_loss

Determine:

1. When the abnormal behavior started.
2. The baseline/before-problem value when available.
3. The worst observed value.
4. The timestamp of the worst value.
5. Whether the condition continued, improved, or recovered.
6. The trend.

For signal_strength:

- values closer to 0 are stronger
- more negative values are weaker

You may use classify_signal_strength for signal values.

Do not investigate operation logs or user sessions.
Do not claim causation.

Your job is to establish WHAT happened and WHEN it happened.

After producing the metric findings, stop.
""",
)


# ============================================================
# 10. OPERATION AGENT
# ============================================================

operation_agent = create_react_agent(
    model=model,
    tools=[get_operation_history],
    name="operation_expert",
    prompt="""
You are a Device Operation Log Specialist.

Your job is to identify router operations around a
problematic date.

Use get_operation_history.

Pay attention to operations that occurred before or
around the identified degradation time.

Examples:

- firmware upgrade
- reboot
- WiFi configuration change
- WAN configuration change
- reset

Report:

- operation
- timestamp
- status
- relevant details

Do NOT claim an operation caused the problem.

Only establish what happened and when.

After producing the operation findings, stop.
""",
)


# ============================================================
# 11. SESSION AGENT
# ============================================================

session_agent = create_react_agent(
    model=model,
    tools=[get_session_history],
    name="session_expert",
    prompt="""
You are a User Session History Specialist.

Your job is to identify user actions around a
problematic date.

Use get_session_history.

Pay attention to actions that happened before or around
the identified degradation time.

Examples:

- firmware upgrade request
- configuration change
- restart request
- speed test
- troubleshooting action

Report:

- action
- timestamp
- status
- user_id when relevant

Do NOT claim that the user action caused the problem.

Only establish what the user did and when.

After producing the session findings, stop.
""",
)


# ============================================================
# 12. DIAGNOSIS AGENT
# ============================================================

diagnosis_agent = create_react_agent(
    model=model,
    tools=[],
    name="diagnosis_expert",
    prompt="""
You are a Router Diagnostic Specialist.

You receive findings from multiple specialist agents.

Possible evidence sources:

- Device Health KPI
- Device KPI
- Raw Device Metric
- Operation Log
- Session History

Your job is to correlate the evidence.

The user may have asked a GENERAL health question or
a SPECIFIC diagnostic question.

For a GENERAL health question:

1. State the overall health.
2. Identify healthy categories.
3. Identify degraded categories.
4. Identify the most important issue.
5. Correlate raw metrics if available.
6. Correlate operations and user actions if available.
7. Explain possible causes only when supported by timing
   and evidence.

For a SPECIFIC question:

Focus on the requested category.

Always distinguish:

FACT
Something directly supported by data.

POSSIBLE_CAUSE
A hypothesis supported by evidence or temporal correlation,
but not proven.

UNKNOWN
Something that cannot be determined from the available data.

IMPORTANT:

Correlation is not causation.

Example:

FACT:
Firmware upgrade occurred at 07:40.

FACT:
Signal degradation started around 08:00.

POSSIBLE_CAUSE:
The firmware upgrade may be related based on timing.

UNKNOWN:
Whether the firmware upgrade actually caused the problem.

Never invent missing values.

Produce a concise technical diagnosis.
""",
)


# ============================================================
# 13. REVIEWER AGENT
# ============================================================

reviewer_agent = create_react_agent(
    model=model,
    tools=[],
    name="reviewer_expert",
    prompt="""
You are a Router Diagnostic Reviewer.

Review the findings and diagnosis against the available
conversation evidence.

Check:

1. Was the correct device analyzed?
2. Was the latest KPI information considered?
3. Were abnormal categories identified correctly?
4. If raw metrics were used, are the timestamps supported?
5. Are the worst values supported by the data?
6. Were operation logs considered when relevant?
7. Were session actions considered when relevant?
8. Were timestamps used when discussing possible causes?
9. Was correlation incorrectly presented as causation?
10. Are FACT, POSSIBLE_CAUSE, and UNKNOWN clearly separated?

If supported, return:

PASS

and a short reason.

If unsupported or incomplete, return:

FAIL

and explain exactly what is missing.

Do not invent new evidence.
""",
)


# ============================================================
# 14. SUPERVISOR
# ============================================================

workflow = create_supervisor(
    [
        health_agent,
        kpi_agent,
        metric_agent,
        operation_agent,
        session_agent,
        diagnosis_agent,
        reviewer_agent,
    ],
    model=model,
    prompt="""
You are the Router Health Diagnostic Supervisor.

Your job is to understand the user's intent and coordinate
the specialist agents.

The user does NOT need to know the internal diagnostic workflow.

============================================================
INTENT TYPES
============================================================

GENERAL HEALTH

Examples:

- "How is ROUTER-001?"
- "How is my router?"
- "Is there anything wrong with my device?"
- "What's the health status?"

For general health:

1. Start with health_expert.
2. Let health_expert identify which categories are abnormal.
3. Investigate only relevant abnormal categories.
4. Use metric_expert when raw measurements are needed.
5. Use operation_expert when operations may provide context.
6. Use session_expert when user actions may provide context.
7. Send the combined evidence to diagnosis_expert.
8. Use reviewer_expert to validate the diagnosis.
9. Finish.

Do not automatically investigate every possible category.

============================================================
SPECIFIC DIAGNOSTIC REQUEST
============================================================

Examples:

- "Why is the signal weak?"
- "Why is the download speed slow?"
- "When did the problem start?"
- "Did the firmware upgrade affect the router?"

For a specific request:

1. Use the relevant specialist.
2. Retrieve deeper evidence only when needed.
3. Use operation/session context when relevant.
4. Send evidence to diagnosis_expert.
5. Use reviewer_expert for multi-source conclusions.
6. Finish.

============================================================
DATA ARCHITECTURE
============================================================

Kafka is NOT queried by the AI.

Kafka data has already been persisted.

device_kpi:
    Daily aggregated health data.

device_metric:
    Raw time-series measurements originating from Kafka.

operation_log:
    Operations performed on the router.

session_history:
    Actions performed or requested by the user.

Use:

device_kpi -> understand recent health/trends.

device_metric -> drill down into WHEN the problem happened.

operation_log -> understand what happened to the device.

session_history -> understand what the user did.

============================================================
DIAGNOSTIC RULES
============================================================

Do not claim causation without evidence.

Always distinguish:

FACT
POSSIBLE_CAUSE
UNKNOWN

Use timestamps when correlating events.

Do not invent values.

Do not repeatedly call the same specialist.

A specialist should normally be used once per investigation.

============================================================
IMPORTANT
============================================================

For a general health request, do NOT automatically assume
the problem is signal strength.

The abnormal category could be:

- WiFi
- signal
- download speed
- upload speed
- latency
- packet loss
- uptime/stability
- firmware/configuration-related behavior

The health assessment determines which area needs investigation.

Finish when the user's question has been adequately answered.
""",
    output_mode="full_history",
)


# ============================================================
# 15. COMPILE
# ============================================================

app = workflow.compile()


def get_user_request() -> str:
    """
    Read user request from command line argument or interactive input.

    Priority:
    1) CLI argument(s): python example_4.py "How is ROUTER-002?"
    2) Interactive prompt
    """

    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()

    request = input("Enter your request: ").strip()

    if request:
        return request

    return "How is ROUTER-002?"


# ============================================================
# 16. TEST CASE
# ============================================================

# Change this to test different scenarios.
#
# Recommended:
#
# "How is ROUTER-001?"
# "How is ROUTER-002?"
# "How is ROUTER-003?"
# "Why is the signal strength of ROUTER-001 weak?"
# "Why is ROUTER-002 slow?"
#
user_request = get_user_request()


# ============================================================
# 17. RUN
# ============================================================

print("\n")
print("=" * 80)
print("USER REQUEST")
print("=" * 80)
print(user_request)

result = app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": user_request,
            }
        ]
    }
)


# ============================================================
# 18. PRINT MESSAGE HISTORY
# ============================================================

print("\n")
print("=" * 80)
print("MESSAGE HISTORY")
print("=" * 80)

for index, message in enumerate(
    result["messages"],
    start=1,
):
    print(f"\n--- MESSAGE {index} ---")
    print("Type:", type(message).__name__)
    print("Name:", getattr(message, "name", None))
    print("Content:", message.content)

    if getattr(message, "tool_calls", None):
        print("Tool calls:", message.tool_calls)


# ============================================================
# 19. FINAL RESULT
# ============================================================

print("\n")
print("=" * 80)
print("FINAL RESULT")
print("=" * 80)
print(result["messages"][-1].content)
