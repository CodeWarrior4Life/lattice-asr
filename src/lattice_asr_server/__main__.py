import uvicorn

from lattice_asr_server.app import create_app


def main() -> None:
    uvicorn.run(create_app(), host="0.0.0.0", port=5556)


if __name__ == "__main__":
    main()
