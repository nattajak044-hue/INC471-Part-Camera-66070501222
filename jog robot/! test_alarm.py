from pydobot.message import Message

def get_alarms_state(device):
    msg = Message()
    msg.id = 20
    msg.ctrl = 0x00
    return device._send_command(msg)

def clear_all_alarms(device):
    msg = Message()
    msg.id = 20
    msg.ctrl = 0x01
    return device._send_command(msg)


if __name__ == "__main__":
    from pydobot import Dobot

    device = Dobot(port='/dev/ttyUSB0', verbose=True)

    resp = get_alarms_state(device)
    print("raw alarm params:", list(resp.params))

    clear_all_alarms(device)

    resp_after = get_alarms_state(device)
    print("raw alarm params after clear:", list(resp_after.params))

    device.close()