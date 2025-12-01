import asyncio

async def forward_frame_to_proctors(session_id: str, frame_data_payload: dict, active_sessions: dict, sio):
    """Emit processed frame payload to faculty room and directly to faculty SIDs as fallback."""
    room_name = f"exam_{session_id}"
    try:
        await sio.emit("receive_frame", frame_data_payload, room=room_name)
    except Exception as e:
        print(f"⚠️ Failed to emit frame to room {room_name}: {e}")

    faculty_list = active_sessions.get(session_id, {}).get('faculty_members', []) or []
    for fac_sid in faculty_list:
        try:
            await sio.emit("receive_frame", frame_data_payload, to=fac_sid)
        except Exception as e:
            print(f"⚠️ Failed to emit frame directly to faculty {fac_sid}: {e}")


async def forward_screen_to_proctors(session_id: str, payload: dict, active_sessions: dict, sio):
    room_name = f"exam_{session_id}"
    try:
        await sio.emit("receive_screen_frame", payload, room=room_name)
    except Exception as e:
        print(f"⚠️ Failed to emit screen frame to room {room_name}: {e}")

    faculty_list = active_sessions.get(session_id, {}).get('faculty_members', []) or []
    for fac_sid in faculty_list:
        try:
            await sio.emit("receive_screen_frame", payload, to=fac_sid)
        except Exception as e:
            print(f"⚠️ Failed to emit screen frame directly to faculty {fac_sid}: {e}")
