import { net } from 'electron';

const HespiProtocols = {
  FILE_LOADER: {
    scheme: 'file-loader',
    privileges: {
      bypassCSP: true,
      // standard: true, // Setting it as standard would not parse absolute paths properly (it seemed to remove the initial '/' and lowercase the first path's letter)
      secure: true,
      supportFetchAPI: true,
    },
    handleFunction: (request) => {
      var fileUrl = 'file://' + request.url.replace('file-loader://', '')
      console.log('Fetching with file-loader: ' + fileUrl);
      return net.fetch(fileUrl)
    }
  },
}

export const registerProtocols = (electronProtocol) => {
  electronProtocol.registerSchemesAsPrivileged(Object.values(HespiProtocols).map((protocol) => {
    console.log(`Registering protocol: ${protocol.scheme}`)
    return {
      scheme: protocol.scheme,
      privileges: protocol.privileges
    }
  }));
}

export const handleProtocols = (electronProtocol) => {
  Object.values(HespiProtocols).forEach((protocol) => {
    console.log(`Registering protocol HANDLE: ${protocol.scheme}`)
    electronProtocol.handle(protocol.scheme, protocol.handleFunction)
  })
}