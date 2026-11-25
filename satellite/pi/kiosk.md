# Pi kiosk: Chromium showing the face page

The face is a local page served by the satellite client on
`http://127.0.0.1:8765/`. It never reaches the relay and holds no token.

## Packages

```sh
sudo apt install python3-aiohttp pipewire pipewire-pulse wireplumber \
                 chromium-browser
```

## Autostart (Pi OS Bookworm, Wayland/labwc)

`~/.config/labwc/autostart`:

```sh
# face kiosk — waits for the client's UI port before launching
until curl -sf http://127.0.0.1:8765/ >/dev/null; do sleep 1; done
chromium-browser --kiosk --ozone-platform=wayland \
  --noerrdialogs --disable-infobars --disable-session-crashed-bubble \
  --check-for-update-interval=31536000 \
  http://127.0.0.1:8765/ &
```

The client unit (`satellite-client.service`) must be enabled first:

```sh
mkdir -p ~/.config/systemd/user
cp ~/satellite/satellite/pi/satellite-client.service ~/.config/systemd/user/
systemctl --user enable --now satellite-client
```

## Display

Touch Display 2 is native portrait (720x1280). Rotate in the compositor, not
in the page — `~/.config/labwc/rc.xml`:

```xml
<output name="DSI-1" transform="90" />
```

Screen blanking off (`~/.config/labwc/autostart`):

```sh
wlr-randr --output DSI-1 --on
gsettings set org.gnome.desktop.session idle-delay 0
```

## Checks

- `curl -s http://127.0.0.1:8765/ | head -1` — the page is served.
- `curl -N http://127.0.0.1:8765/events` — one `data:` line per state change.
- `curl -X POST http://127.0.0.1:8765/tap` — should report
  `{"ok":true,"action":"start","state":"listening"}` when idle and connected.
- Tap the screen: the circle turns green (`listening`).
