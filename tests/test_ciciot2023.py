from ids.data.ciciot2023 import attack_to_category


def test_attack_family_mapping():
    assert attack_to_category("BenignTraffic") == "Benign"
    assert attack_to_category("DDoS-ICMP_Flood") == "DDoS"
    assert attack_to_category("DoS-SYN_Flood") == "DoS"
    assert attack_to_category("Mirai-greip_flood") == "Mirai"
    assert attack_to_category("Recon-PortScan") == "Recon"
    assert attack_to_category("MITM-ArpSpoofing") == "Spoofing"
    assert attack_to_category("DictionaryBruteForce") == "BruteForce"
    assert attack_to_category("SqlInjection") == "WebBased"
